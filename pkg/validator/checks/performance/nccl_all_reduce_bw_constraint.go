// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package performance

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/NVIDIA/eidos/pkg/defaults"
	eidosErrors "github.com/NVIDIA/eidos/pkg/errors"
	"github.com/NVIDIA/eidos/pkg/recipe"
	"github.com/NVIDIA/eidos/pkg/validator/checks"
	"github.com/NVIDIA/eidos/pkg/validator/helper"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/yaml"
)

const (
	trainJobTimeout      = 30 * time.Minute
	launcherPodTimeout   = 5 * time.Minute
	runtimeTemplateFile  = "testdata/runtime.yaml"
	trainJobTemplateFile = "testdata/trainjob.yaml"
	testType             = "all_reduce_perf"
	minMessageSize       = "8G"
	maxMessageSize       = "16G"
)

func init() {
	// Register this constraint validator
	checks.RegisterConstraintValidator(&checks.ConstraintValidator{
		Pattern:     "nccl-all-reduce-bw",
		Description: "Verify NCCL All Reduce Bus Bandwidth within 10% of threshold",
		Func:        validateNcclAllReduceBw,
		TestName:    "TestNcclAllReduceBw",
		Phase:       "performance",
	})
}

// validateNcclAllReduceBw validates NCCL All Reduce bandwidth by running a TrainJob.
// It applies the runtime and trainjob YAML templates, waits for completion,
// and extracts bandwidth metrics from the launcher pod logs.
// Returns actual bandwidth value, whether it passed the threshold, and any error.
func validateNcclAllReduceBw(ctx *checks.ValidationContext, constraint recipe.Constraint, t *testing.T) (string, bool, error) {
	t.Log("Starting NCCL All Reduce bandwidth validation")

	// Extract threshold from constraint
	threshold, err := parseThreshold(constraint.Value)
	if err != nil {
		return "", false, eidosErrors.Wrap(eidosErrors.ErrCodeInvalidRequest, "invalid threshold", err)
	}
	t.Logf("Target bandwidth threshold: %.2f GB/s (with 10%% tolerance)", threshold)

	// Determine GPU configuration from snapshot
	gpuConfig, err := determineGPUConfig(ctx, t)
	if err != nil {
		return "", false, eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to determine GPU configuration", err)
	}
	t.Logf("GPU Configuration: %d nodes, %d GPUs/node, %d total GPUs",
		gpuConfig.WorkerCount, gpuConfig.GPUCountPerNode, gpuConfig.TotalGPUCount)

	// Create dynamic client for CRD operations
	dynamicClient, err := dynamic.NewForConfig(ctx.RESTConfig)
	if err != nil {
		return "", false, eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to create dynamic client", err)
	}

	// Apply runtime and trainjob resources
	if err := applyNCCLResources(ctx, dynamicClient, gpuConfig, t); err != nil {
		return "", false, eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to apply NCCL resources", err)
	}

	// Ensure cleanup
	defer cleanupNCCLResources(ctx, dynamicClient, gpuConfig.Namespace, t)

	// Create pod helper for launcher pod operations
	podHelper := &helper.PodLifecycle{
		ClientSet:  ctx.Clientset,
		RESTConfig: ctx.RESTConfig,
		Namespace:  ctx.Namespace,
		T:          t,
	}

	// Wait for launcher pod and get logs
	logs, err := waitForLauncherPodAndGetLogs(ctx, podHelper, t)
	if err != nil {
		return "", false, eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to get launcher logs", err)
	}

	// Parse bandwidth from logs
	bandwidth, err := parseBandwidthFromLogs(logs)
	if err != nil {
		return logs, false, eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to parse bandwidth from logs", err)
	}

	t.Logf("Measured bandwidth: %.2f GB/s", bandwidth)

	// Check if bandwidth meets threshold (within 10% tolerance)
	passed := bandwidth >= (threshold * 0.9)
	actualValue := fmt.Sprintf("%.2f GB/s", bandwidth)

	if passed {
		t.Logf("✓ Bandwidth validation passed: %.2f GB/s >= %.2f GB/s (90%% of threshold)",
			bandwidth, threshold*0.9)
	} else {
		t.Logf("✗ Bandwidth validation failed: %.2f GB/s < %.2f GB/s (90%% of threshold)",
			bandwidth, threshold*0.9)
	}

	return actualValue, passed, nil
}

// gpuConfiguration holds GPU node and count information
type gpuConfiguration struct {
	WorkerCount     int
	GPUCountPerNode int
	TotalGPUCount   int
	Namespace       string
}

// parseThreshold extracts the numeric threshold value from constraint
func parseThreshold(value string) (float64, error) {
	// Remove units and parse (e.g., "450 GB/s" -> 450)
	numStr := strings.TrimSpace(value)
	numStr = strings.Split(numStr, " ")[0]

	threshold, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid threshold format: %w", err)
	}

	return threshold, nil
}

// determineGPUConfig analyzes the snapshot to determine GPU node configuration
func determineGPUConfig(ctx *checks.ValidationContext, t *testing.T) (*gpuConfiguration, error) {
	t.Log("Analyzing GPU node configuration...")

	// Find schedulable GPU nodes
	gpuNodes, err := helper.FindSchedulableGpuNodes(ctx)
	if err != nil {
		return nil, eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to find GPU nodes", err)
	}

	if len(gpuNodes) == 0 {
		return nil, eidosErrors.New(eidosErrors.ErrCodeInternal, "no schedulable GPU nodes found")
	}

	t.Logf("Found %d GPU node(s)", len(gpuNodes))

	// Get GPU count from first node (assuming homogeneous cluster)
	firstNode := gpuNodes[0]
	gpuResource := v1.ResourceName("nvidia.com/gpu")
	gpuQuantity := firstNode.Status.Allocatable[gpuResource]
	gpuCountPerNode := int(gpuQuantity.Value())

	if gpuCountPerNode == 0 {
		return nil, eidosErrors.New(eidosErrors.ErrCodeInternal, "no GPUs found on nodes")
	}

	totalGPUs := len(gpuNodes) * gpuCountPerNode

	return &gpuConfiguration{
		WorkerCount:     len(gpuNodes),
		GPUCountPerNode: gpuCountPerNode,
		TotalGPUCount:   totalGPUs,
		Namespace:       ctx.Namespace,
	}, nil
}

// applyNCCLResources applies the runtime and trainjob YAML files with template substitution using dynamic client
func applyNCCLResources(ctx *checks.ValidationContext, dynamicClient dynamic.Interface, config *gpuConfiguration, t *testing.T) error {
	t.Log("Applying NCCL test resources...")

	templateData := map[string]string{
		"NAMESPACE":          config.Namespace,
		"WORKER_COUNT":       strconv.Itoa(config.WorkerCount),
		"GPU_COUNT_PER_NODE": strconv.Itoa(config.GPUCountPerNode),
		"GPU_COUNT":          strconv.Itoa(config.TotalGPUCount),
		"TEST_TYPE":          testType,
		"MIN_MESSAGE_SIZE":   minMessageSize,
		"MAX_MESSAGE_SIZE":   maxMessageSize,
	}

	// Get the directory containing this source file
	_, filename, _, _ := runtime.Caller(0)
	baseDir := filepath.Dir(filename)

	// Define GVRs for TrainingRuntime and TrainJob
	trainingRuntimeGVR := schema.GroupVersionResource{
		Group:    "trainer.kubeflow.org",
		Version:  "v1alpha1",
		Resource: "trainingruntimes",
	}

	trainJobGVR := schema.GroupVersionResource{
		Group:    "trainer.kubeflow.org",
		Version:  "v1alpha1",
		Resource: "trainjobs",
	}

	// Apply runtime first
	runtimePath := filepath.Join(baseDir, runtimeTemplateFile)
	if err := applyYAMLWithDynamicClient(ctx.Context, dynamicClient, trainingRuntimeGVR, config.Namespace, runtimePath, templateData); err != nil {
		return eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to apply training runtime", err)
	}
	t.Log("✓ Applied TrainingRuntime")

	// Apply trainjob
	trainJobPath := filepath.Join(baseDir, trainJobTemplateFile)
	if err := applyYAMLWithDynamicClient(ctx.Context, dynamicClient, trainJobGVR, config.Namespace, trainJobPath, templateData); err != nil {
		return eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to apply train job", err)
	}
	t.Log("✓ Applied TrainJob")

	return nil
}

// applyYAMLWithDynamicClient reads a YAML template, performs substitution, and applies it using dynamic client
func applyYAMLWithDynamicClient(ctx context.Context, dynamicClient dynamic.Interface, gvr schema.GroupVersionResource, namespace, templatePath string, data map[string]string) error {
	content, err := os.ReadFile(templatePath)
	if err != nil {
		return eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to read template", err)
	}

	// Perform template substitution
	yamlContent := string(content)
	for key, value := range data {
		yamlContent = strings.ReplaceAll(yamlContent, "${"+key+"}", value)
	}

	// Parse YAML to unstructured object
	obj := &unstructured.Unstructured{}
	if err := yaml.Unmarshal([]byte(yamlContent), obj); err != nil {
		return eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to parse YAML", err)
	}

	// Apply with timeout
	applyCtx, cancel := context.WithTimeout(ctx, defaults.DiagnosticTimeout)
	defer cancel()

	// Create the resource
	_, err = dynamicClient.Resource(gvr).Namespace(namespace).Create(applyCtx, obj, metav1.CreateOptions{})
	if err != nil {
		return eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to create resource", err)
	}

	return nil
}

// waitForLauncherPodAndGetLogs waits for the launcher pod to be created and retrieves logs
func waitForLauncherPodAndGetLogs(ctx *checks.ValidationContext, podHelper *helper.PodLifecycle, t *testing.T) (string, error) {
	t.Log("Waiting for launcher pod to be created...")

	// Wait for launcher pod to be created (pattern: nccl-all-reduce-tj-launcher-*)
	launcherPod, err := waitForPodByLabelSelector(
		ctx.Context,
		ctx.Clientset,
		ctx.Namespace,
		"trainer.kubeflow.org/job-name=nccl-all-reduce-tj,trainer.kubeflow.org/job-role=launcher",
		launcherPodTimeout,
		t,
	)
	if err != nil {
		return "", eidosErrors.Wrap(eidosErrors.ErrCodeTimeout, "failed to find launcher pod", err)
	}

	t.Logf("Found launcher pod: %s", launcherPod.Name)

	// Wait for pod to complete using helper method
	err = podHelper.WaitForPodSuccess(ctx.Context, launcherPod, trainJobTimeout)
	if err != nil {
		// Get logs even if pod failed for debugging
		t.Log("Pod did not succeed, retrieving logs for debugging...")
		logs, _ := podHelper.GetPodLogs(ctx.Context, launcherPod)
		return logs, eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "pod failed to complete successfully", err)
	}

	// Get logs from completed pod using helper method
	t.Log("Retrieving logs from successful pod...")
	logs, err := podHelper.GetPodLogs(ctx.Context, launcherPod)
	if err != nil {
		return "", eidosErrors.Wrap(eidosErrors.ErrCodeInternal, "failed to get pod logs", err)
	}

	return logs, nil
}

// waitForPodByLabelSelector waits for a pod matching the label selector to be created
func waitForPodByLabelSelector(ctx context.Context, clientset kubernetes.Interface, namespace, labelSelector string, timeout time.Duration, t *testing.T) (*v1.Pod, error) {
	t.Logf("Polling for pod with selector: %s", labelSelector)

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ticker := time.NewTicker(defaults.PodPollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, eidosErrors.Wrap(eidosErrors.ErrCodeTimeout, "timeout waiting for pod", ctx.Err())
		case <-ticker.C:
			pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
				LabelSelector: labelSelector,
			})
			if err != nil {
				t.Logf("Error listing pods: %v, retrying...", err)
				continue
			}
			if len(pods.Items) > 0 {
				return &pods.Items[0], nil
			}
			t.Log("Pod not found yet, continuing to poll...")
		}
	}
}

// parseBandwidthFromLogs extracts the bus bandwidth value from NCCL test logs
func parseBandwidthFromLogs(logs string) (float64, error) {
	// NCCL test output format example:
	// #       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
	// #        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
	//  17179869184    4294967296     float     sum      -1   123456   139.2   450.3      0   123456   139.2   450.3      0

	// Look for the line with the max message size (16G = 17179869184 bytes)
	re := regexp.MustCompile(`\s+17179869184\s+\d+\s+\w+\s+\w+\s+-?\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)`)
	matches := re.FindStringSubmatch(logs)

	if len(matches) < 2 {
		return 0, fmt.Errorf("could not find bandwidth value in logs")
	}

	bandwidth, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse bandwidth value: %w", err)
	}

	return bandwidth, nil
}

// cleanupNCCLResources removes the trainjob and runtime resources using dynamic client
func cleanupNCCLResources(ctx *checks.ValidationContext, dynamicClient dynamic.Interface, namespace string, t *testing.T) {
	t.Log("Cleaning up NCCL test resources...")

	cleanupCtx, cancel := context.WithTimeout(context.Background(), defaults.DiagnosticTimeout)
	defer cancel()

	// Define GVRs
	trainJobGVR := schema.GroupVersionResource{
		Group:    "trainer.kubeflow.org",
		Version:  "v1alpha1",
		Resource: "trainjobs",
	}

	trainingRuntimeGVR := schema.GroupVersionResource{
		Group:    "trainer.kubeflow.org",
		Version:  "v1alpha1",
		Resource: "trainingruntimes",
	}

	// Delete trainjob
	err := dynamicClient.Resource(trainJobGVR).Namespace(namespace).Delete(cleanupCtx, "nccl-all-reduce-tj", metav1.DeleteOptions{})
	if err != nil {
		t.Logf("Warning: Failed to delete TrainJob: %v", err)
	} else {
		t.Log("✓ Deleted TrainJob")
	}

	// Delete runtime
	err = dynamicClient.Resource(trainingRuntimeGVR).Namespace(namespace).Delete(cleanupCtx, "nccl-all-reduce-runtime", metav1.DeleteOptions{})
	if err != nil {
		t.Logf("Warning: Failed to delete TrainingRuntime: %v", err)
	} else {
		t.Log("✓ Deleted TrainingRuntime")
	}
}
