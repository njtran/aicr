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
	"testing"

	"github.com/NVIDIA/eidos/pkg/recipe"
	"github.com/NVIDIA/eidos/pkg/validator/checks"
)

func TestValidateNcclAllReduceBw(t *testing.T) {
	tests := []struct {
		name       string
		setup      func() *checks.ValidationContext
		constraint recipe.Constraint
		wantActual string
		wantPassed bool
		wantErr    bool
	}{
		{
			name: "constraint satisfied",
			setup: func() *checks.ValidationContext {
				return &checks.ValidationContext{
					Context: context.Background(),
					// TODO: Add mock clientset if needed
					// Clientset: fake.NewSimpleClientset(...),
				}
			},
			constraint: recipe.Constraint{
				Name:  "nccl-all-reduce-bw",
				Value: "expected-value",
			},
			wantActual: "not-implemented",
			wantPassed: true,
			wantErr:    false,
		},
		// TODO: Add constraint failure test cases when implementation is complete
		// {
		// 	name: "constraint not satisfied",
		// 	setup: func() *checks.ValidationContext {
		// 		return &checks.ValidationContext{
		// 			Context: context.Background(),
		// 			// Setup context that should cause constraint to fail
		// 		}
		// 	},
		// 	constraint: recipe.Constraint{
		// 		Name:  "nccl-all-reduce-bw",
		// 		Value: "different-value",
		// 	},
		// 	wantActual: "actual-value",
		// 	wantPassed: false,
		// 	wantErr:    false,
		// },
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := tt.setup()
			actual, passed, err := validateNcclAllReduceBw(ctx, tt.constraint, t)

			if (err != nil) != tt.wantErr {
				t.Errorf("validateNcclAllReduceBw() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			if actual != tt.wantActual {
				t.Errorf("validateNcclAllReduceBw() actual = %v, want %v", actual, tt.wantActual)
			}

			if passed != tt.wantPassed {
				t.Errorf("validateNcclAllReduceBw() passed = %v, want %v", passed, tt.wantPassed)
			}
		})
	}
}

func TestValidateNcclAllReduceBwRegistration(t *testing.T) {
	// Verify the constraint validator is registered
	validator, ok := checks.GetConstraintValidator("nccl-all-reduce-bw")
	if !ok {
		t.Fatal("nccl-all-reduce-bw constraint validator not registered")
	}

	if validator.Pattern != "nccl-all-reduce-bw" {
		t.Errorf("Pattern = %v, want nccl-all-reduce-bw", validator.Pattern)
	}

	if validator.Description == "" {
		t.Error("Description is empty")
	}

	if validator.TestName == "" {
		t.Error("TestName is empty")
	}
}
