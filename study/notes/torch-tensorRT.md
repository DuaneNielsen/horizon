###building bazel from source

copy the below to a shell script and execute

```bash
#!/bin/bash
#
# Reference: https://docs.bazel.build/versions/master/install-ubuntu.html#install-with-installer-ubuntu

BAZEL_VERSION='3.7.1'

set -e

folder=${HOME}/delete
mkdir -p $folder

echo "** Install requirements"
sudo apt-get install -y pkg-config zip g++ zlib1g-dev unzip
sudo apt-get install -y openjdk-8-jdk

echo "** Download bazel-0.15.2 sources"
cd $folder
if [ ! -f bazel-0.15.2-dist.zip ]; then
  wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-dist.zip
fi

echo "** Build and install bazel-0.15.2"
unzip bazel-${BAZEL_VERSION}-dist.zip -d bazel-${BAZEL_VERSION}-dist
cd bazel-${BAZEL_VERSION}-dist
```

```bash
vi ~/delete/bazel-3.7.1-dist/scripts/bootstrap/compile.sh
```

look for the line

```bash
  run "${JAVAC}" -classpath "${classpath}" -sourcepath "${sourcepath}" \
      -d "${output}/classes" -source "$JAVA_VERSION" -target "$JAVA_VERSION" \
      -encoding UTF-8 ${BAZEL_JAVAC_OPTS} "@${paramfile}" -J-Xmx1848M
```

and add xJ-Xmx1848, like so...

```bash
  run "${JAVAC}" -classpath "${classpath}" -sourcepath "${sourcepath}" \
      -d "${output}/classes" -source "$JAVA_VERSION" -target "$JAVA_VERSION" \
      -encoding UTF-8 ${BAZEL_JAVAC_OPTS} "@${paramfile}" -J-Xmx1848M
```

then compile using local java

```bash
cd ~/delete/bazel-3.7.1-dist/
env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
sudo cp output/bazel /usr/local/bin
bazel help
```



torch_tensorrt

write the below text to ~/WORKSPACE.patch

```text
diff --git a/WORKSPACE b/WORKSPACE
index 2779e93c..8ca2dcd4 100644
--- a/WORKSPACE
+++ b/WORKSPACE
@@ -41,7 +41,7 @@ local_repository(
 new_local_repository(
     name = "cuda",
     build_file = "@//third_party/cuda:BUILD",
-    path = "/usr/local/cuda-11.3/",
+    path = "/usr/local/cuda-10.2/",
 )
 
 new_local_repository(
@@ -53,45 +53,45 @@ new_local_repository(
 # Tarballs and fetched dependencies (default - use in cases when building from precompiled bin and tarballs)
 #############################################################################################################
 
-http_archive(
-    name = "libtorch",
-    build_file = "@//third_party/libtorch:BUILD",
-    sha256 = "8d9e829ce9478db4f35bdb7943308cf02e8a2f58cf9bb10f742462c1d57bf287",
-    strip_prefix = "libtorch",
-    urls = ["https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip"],
-)
+#http_archive(
+#    name = "libtorch",
+#    build_file = "@//third_party/libtorch:BUILD",
+#    sha256 = "8d9e829ce9478db4f35bdb7943308cf02e8a2f58cf9bb10f742462c1d57bf287",
+#    strip_prefix = "libtorch",
+#    urls = ["https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip"],
+#)
 
-http_archive(
-    name = "libtorch_pre_cxx11_abi",
-    build_file = "@//third_party/libtorch:BUILD",
-    sha256 = "90159ecce3ff451f3ef3f657493b6c7c96759c3b74bbd70c1695f2ea2f81e1ad",
-    strip_prefix = "libtorch",
-    urls = ["https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.11.0%2Bcu113.zip"],
-)
+#http_archive(
+#    name = "libtorch_pre_cxx11_abi",
+#    build_file = "@//third_party/libtorch:BUILD",
+#    sha256 = "90159ecce3ff451f3ef3f657493b6c7c96759c3b74bbd70c1695f2ea2f81e1ad",
+#    strip_prefix = "libtorch",
+#    urls = ["https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.11.0%2Bcu113.zip"],
+#)
 
 # Download these tarballs manually from the NVIDIA website
 # Either place them in the distdir directory in third_party and use the --distdir flag
 # or modify the urls to "file:///<PATH TO TARBALL>/<TARBALL NAME>.tar.gz
 
-http_archive(
-    name = "cudnn",
-    build_file = "@//third_party/cudnn/archive:BUILD",
-    sha256 = "0e5d2df890b9967efa6619da421310d97323565a79f05a1a8cb9b7165baad0d7",
-    strip_prefix = "cuda",
-    urls = [
-        "https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.4/11.4_20210831/cudnn-11.4-linux-x64-v8.2.4.15.tgz",
-    ],
-)
+#http_archive(
+#    name = "cudnn",
+#    build_file = "@//third_party/cudnn/archive:BUILD",
+#    sha256 = "0e5d2df890b9967efa6619da421310d97323565a79f05a1a8cb9b7165baad0d7",
+#    strip_prefix = "cuda",
+#    urls = [
+#        "https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.4/11.4_20210831/cudnn-11.4-linux-x64-v8.2.4.15.tgz",
+#    ],
+#)
 
-http_archive(
-    name = "tensorrt",
-    build_file = "@//third_party/tensorrt/archive:BUILD",
-    sha256 = "826180eaaecdf9a7e76116855b9f1f3400ea9b06e66b06a3f6a0747ba6f863ad",
-    strip_prefix = "TensorRT-8.2.4.2",
-    urls = [
-        "https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.4/tars/tensorrt-8.2.4.2.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz",
-    ],
-)
+#http_archive(
+#    name = "tensorrt",
+#    build_file = "@//third_party/tensorrt/archive:BUILD",
+#    sha256 = "826180eaaecdf9a7e76116855b9f1f3400ea9b06e66b06a3f6a0747ba6f863ad",
+#    strip_prefix = "TensorRT-8.2.4.2",
+#    urls = [
+#        "https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.4/tars/tensorrt-8.2.4.2.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz",
+#    ],
+#)
 
 ####################################################################################
 # Locally installed dependencies (use in cases of custom dependencies or aarch64)
@@ -104,29 +104,29 @@ http_archive(
 # x86_64 python distribution. If using NVIDIA's version just point to the root of the package
 # for both versions here and do not use --config=pre-cxx11-abi
 
-#new_local_repository(
-#    name = "libtorch",
-#    path = "/usr/local/lib/python3.6/dist-packages/torch",
-#    build_file = "third_party/libtorch/BUILD"
-#)
+new_local_repository(
+    name = "libtorch",
+    path = "/usr/local/lib/python3.6/dist-packages/torch",
+    build_file = "third_party/libtorch/BUILD"
+)
 
-#new_local_repository(
-#    name = "libtorch_pre_cxx11_abi",
-#    path = "/usr/local/lib/python3.6/dist-packages/torch",
-#    build_file = "third_party/libtorch/BUILD"
-#)
+new_local_repository(
+    name = "libtorch_pre_cxx11_abi",
+    path = "/usr/local/lib/python3.6/dist-packages/torch",
+    build_file = "third_party/libtorch/BUILD"
+)
 
-#new_local_repository(
-#    name = "cudnn",
-#    path = "/usr/",
-#    build_file = "@//third_party/cudnn/local:BUILD"
-#)
+new_local_repository(
+    name = "cudnn",
+    path = "/usr/",
+    build_file = "@//third_party/cudnn/local:BUILD"
+)
 
-#new_local_repository(
-#   name = "tensorrt",
-#   path = "/usr/",
-#   build_file = "@//third_party/tensorrt/local:BUILD"
-#)
+new_local_repository(
+   name = "tensorrt",
+   path = "/usr/",
+   build_file = "@//third_party/tensorrt/local:BUILD"
+)
 
 # #########################################################################
 # # Testing Dependencies (optional - comment out on aarch64)

```


```bash
git clone https://github.com/NVIDIA/Torch-TensorRT.git
cd Torch-TensorRT.git
git apply ~/WORKSPACE.patch
bazel build //:libtorchtrt --platforms //toolchains:jetpack_4.6 --nokeep_state_after_build --discard_analysis_cache --notrack_incremental_state --jobs 1
```