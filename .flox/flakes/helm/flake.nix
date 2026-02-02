# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Helm 4.x flake - builds from git source
#
# nixpkgs has not yet merged Helm 4.x support:
# https://github.com/NixOS/nixpkgs/pull/461007
#
# We build from git source to ensure we have the exact version specified
# in .versions.yaml.

{
  description = "Helm 4.x built from source";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Helm version from .versions.yaml (v4.1.0 -> 4.1.0)
        helmVersion = "4.1.0";
        helmSrc = pkgs.fetchFromGitHub {
          owner = "helm";
          repo = "helm";
          rev = "v${helmVersion}";
          hash = "sha256-evoBc6gXlk6opJCQ1Byi84j2AdpaASmRbxBnQOcxSLU=";
        };

        helm = pkgs.buildGoModule {
          pname = "helm";
          version = helmVersion;
          src = helmSrc;

          # Let Go fetch dependencies (vendor dir is out of sync)
          vendorHash = "sha256-JVQdA7R5rhE0bTcfh3dFat146bQXJaIWGy9D7sv531w=";

          ldflags = [
            "-s" "-w"
            "-X helm.sh/helm/v4/internal/version.version=v${helmVersion}"
            "-X helm.sh/helm/v4/internal/version.gitCommit=${helmSrc.rev}"
            "-X helm.sh/helm/v4/internal/version.gitTreeState=clean"
          ];

          subPackages = [ "cmd/helm" ];

          # Skip tests - they require a writable home directory
          doCheck = false;

          meta = with pkgs.lib; {
            description = "Kubernetes package manager";
            homepage = "https://helm.sh/";
            license = licenses.asl20;
            mainProgram = "helm";
          };
        };
      in
      {
        packages = {
          inherit helm;
          default = helm;
        };
      }
    );
}
