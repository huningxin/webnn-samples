"use strict";

import * as utils from "../common/utils.js";
import { buildWebGL2Pipeline } from "./lib/webgl2/webgl2Pipeline.js";
import * as ui from "../common/ui.js";
import { WebnnSelfieSegmenterGeneral } from "./webnn_selfie_segmenter_general.js";
import { WebnnSelfieSegmenterLandscape } from "./webnn_selfie_segmenter_landscape.js";
import { SelfieSegmenterNhwc } from "./selfie_segmenter_nhwc.js"
import { SelfieSegmenter } from "./selfie_segmenter.js"

const imgElement = document.getElementById("feedElement");
imgElement.src = "./images/test.jpg";
const camElement = document.getElementById("feedMediaElement");
const outputCanvas = document.getElementById("outputCanvas");
let rafReq;
let isFirstTimeLoad = true;
let inputType = "image";
let stream = null;
let sess;
let wnnModel;
let loadTime = 0;
let computeTime = 0;
let outputBuffer;
let modelChanged = false;
let backgroundImageSource = document.getElementById("blur-img");
let backgroundType = "blur"; // 'none', 'blur', 'image'
let modelType = "webnn"; // webnn or ort
let deviceType = "";
let backend = "webnn";
let startRun;
const numMinutes = 1;
let count = 0;
let computeTimeArray = [];
let perfTest = false;
let resolutionType = "general";
const inputOptions = {
  mean: [127.5, 127.5, 127.5],
  std: [127.5, 127.5, 127.5],
  scaledFlag: false,
  inputResolution: [256, 144],
};

const disabledSelectors = ["#tabs > li", ".btn"];

let gpuBuffer = null;
let gpuInputTensor = null;
function uploadToGPU(gpuBuffer, inputBuffer) {
  const stagingBuffer = ort.env.webgpu.device.createBuffer({
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    size: inputBuffer.byteLength,
    mappedAtCreation: true,
  });
  const arrayBuffer = stagingBuffer.getMappedRange();
  new Uint8Array(arrayBuffer).set(
    new Uint8Array(
      inputBuffer.buffer,
      inputBuffer.byteOffset,
      inputBuffer.byteLength
    )
  );
  stagingBuffer.unmap();
  const encoder = ort.env.webgpu.device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    stagingBuffer,
    0,
    gpuBuffer,
    0,
    inputBuffer.byteLength
  );
  ort.env.webgpu.device.queue.submit([encoder.finish()]);
  stagingBuffer.destroy();
}

function createGpuTensorForInput(inputBuffer) {
  if (
    !gpuBuffer ||
    gpuBuffer.size != Math.ceil(inputBuffer.byteLength / 16) * 16
  ) {
    gpuBuffer = ort.env.webgpu.device.createBuffer({
      // eslint-disable-next-line no-bitwise
      usage:
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.STORAGE,
      size: Math.ceil(inputBuffer.byteLength / 16) * 16,
    });
  }
  uploadToGPU(gpuBuffer, inputBuffer);

  if (!gpuInputTensor || gpuInputTensor.dims[1] != inputOptions.inputShape[1]) {
    gpuInputTensor = ort.Tensor.fromGpuBuffer(gpuBuffer, {
      dataType: "float32",
      dims: inputOptions.inputShape,
      dispose: () => gpuBuffer.destroy(),
    });
  }
}

async function compute(modelType, inputBuffer) {
  // console.time('compute function');
  let outputData;
  if (modelType == "ort") {
    const feed = {};

    if (backend.includes("webgpu")) {
      createGpuTensorForInput(inputBuffer);
      feed["input"] = gpuInputTensor;
    } else {
      feed["input"] = new ort.Tensor(
        "float32",
        inputBuffer,
        inputOptions.inputShape
      );
    }
    const result = await sess.run(feed);
    if (backend.includes("webgpu")) {
      outputData = await result["segment_back"].getData();
    } else {
      outputData = result["segment_back"].cpuData;
    }
  } else {
    // outputData = await wnnModel.compute(inputBuffer);
    outputData = (await wnnModel.run({'input': inputBuffer}))['segment_back'];
  }
  // console.timeEnd('compute function');
  return outputData;
}

$(document).ready(async () => {
  $(".icdisplay").hide();
  $("#landscape").click();
  ort.env.wasm.numThreads = 4;
});

$("#backend").on("change", async (e) => {
  modelChanged = true;
  backend = $(e.target).find(":selected").val();
  modelType = backend.includes("ort") ? "ort" : "webnn";
  if (!backend.includes("webnn") && deviceType != "") {
    $(`#${deviceType}`).parent().removeClass("active");
  }
  if (inputType === "camera") utils.stopCameraStream(rafReq, stream);
  await main();
});

$("#resolutionType").on("change", async (e) => {
  modelChanged = true;
  resolutionType = $(e.target).attr("id");
  if (resolutionType === "general") {
    inputOptions.inputResolution = [256, 256];
  } else {
    inputOptions.inputResolution = [256, 144];
  }
  if (inputType === "camera") utils.stopCameraStream(rafReq, stream);
  await main();
});

$("#deviceTypeBtns").on("change", async (e) => {
  modelChanged = true;
  deviceType = $(e.target).attr("id");
  if (inputType === "camera") utils.stopCameraStream(rafReq, stream);
  await main();
});

// Click trigger to do inference with <img> element
$("#img").click(async () => {
  if (inputType === "camera") utils.stopCameraStream(rafReq, stream);
  inputType = "image";
  $("#pickimage").show();
  $(".shoulddisplay").hide();
  await main();
});

$("#imageFile").change((e) => {
  const files = e.target.files;
  if (files.length > 0) {
    $("#feedElement").removeAttr("height");
    $("#feedElement").removeAttr("width");
    imgElement.src = URL.createObjectURL(files[0]);
  }
});

$("#feedElement").on("load", async () => {
  await main();
});

// Click trigger to do inference with <video> media element
$("#cam").click(async () => {
  inputType = "camera";
  $("#pickimage").hide();
  $(".shoulddisplay").hide();
  await main();
});

$("#gallery .gallery-item").click(async (e) => {
  $("#gallery .gallery-item").removeClass("hl");
  $(e.target).parent().addClass("hl");
  const backgroundTypeId = $(e.target).attr("id");
  backgroundImageSource = document.getElementById(backgroundTypeId);
  if (backgroundTypeId === "no-img") {
    backgroundType = "none";
  } else if (backgroundTypeId === "blur-img") {
    backgroundType = "blur";
  } else {
    backgroundType = "image";
  }
  const srcElement = inputType == "image" ? imgElement : camElement;
  await drawOutput(outputBuffer, srcElement);
});

/**
 * This method is used to render live camera tab.
 */
async function renderCamStream() {
  if (!stream.active) return;
  // If the video element's readyState is 0, the video's width and height are 0.
  // So check the readState here to make sure it is greater than 0.
  if (camElement.readyState === 0) {
    rafReq = requestAnimationFrame(renderCamStream);
    return;
  }
  const inputCanvas = utils.getVideoFrame(camElement);
  const inputBuffer = utils.getInputTensor(camElement, inputOptions);

  if (perfTest) {
    if (performance.now() - startRun <= 1000 * 60 * numMinutes) {
      // only record for 1 minute
      count++;
      const start = performance.now();
      outputBuffer = await compute(modelType, inputBuffer);
      computeTime = performance.now() - start;
      console.log(`  compute time ${count}: ${computeTime.toFixed(2)} ms`);
      if (count > 3 && computeTime !== 0) {
        // skip first 3 inferences, treat as warmup
        computeTimeArray.push(computeTime);
      }
    } else {
      if (computeTimeArray.length > 0) {
        computeTime = utils.getMedianValue(computeTimeArray);
        const result = `median compute time: ${computeTime.toFixed(
          2
        )} ms, run times: ${count}`;
        console.log(result);
        alert(result);
        computeTimeArray = [];
        count = 0;
      }
    }
  } else {
    const start = performance.now();
    outputBuffer = await compute(modelType, inputBuffer);
    computeTime = performance.now() - start;
    console.log(`  done in ${computeTime.toFixed(2)} ms.`);
  }

  showPerfResult();
  await drawOutput(outputBuffer, inputCanvas);
  $("#fps").text(`${(1000 / computeTime).toFixed(0)} FPS`);
  rafReq = requestAnimationFrame(renderCamStream);
}

async function drawOutput(outputBuffer, srcElement) {
  outputCanvas.width = srcElement.width;
  outputCanvas.height = srcElement.height;
  const pipeline = buildWebGL2Pipeline(
    srcElement,
    backgroundImageSource,
    backgroundType,
    inputOptions.inputResolution,
    outputCanvas,
    outputBuffer
  );
  const postProcessingConfig = {
    smoothSegmentationMask: true,
    jointBilateralFilter: { sigmaSpace: 1, sigmaColor: 0.1 },
    coverage: [0.5, 0.75],
    lightWrapping: 0.3,
    blendMode: "screen",
  };
  pipeline.updatePostProcessingConfig(postProcessingConfig);
  await pipeline.render();
}

function showPerfResult(medianComputeTime = undefined) {
  $("#loadTime").html(`${loadTime.toFixed(2)} ms`);
  if (medianComputeTime !== undefined) {
    $("#computeLabel").html("Median inference time:");
    $("#computeTime").html(`${medianComputeTime.toFixed(2)} ms`);
  } else {
    $("#computeLabel").html("Inference time:");
    $("#computeTime").html(`${computeTime.toFixed(2)} ms`);
  }
}

export async function main() {
  try {
    if (deviceType === "" && backend.includes("webnn")) return;
    ui.handleClick(disabledSelectors, true);
    if (isFirstTimeLoad) $("#hint").hide();
    const numRuns = utils.getUrlParams()[0];
    // Running test for 1 minute
    perfTest = !!utils.getUrlParams()[3];
    // Only do load() when model first time loads and
    // there's new model or delegate choosed
    if (isFirstTimeLoad || modelChanged) {
      modelChanged = false;
      isFirstTimeLoad = false;
      // UI shows model loading progress
      await ui.showProgressComponent("current", "pending", "pending");
      const start = performance.now();
      if (modelType == "ort") {
        const provider = backend.split("-")[1];
        console.log(
          `- Loading ORT model, provider: [${provider}], deviceType: [${deviceType}]`
        );
        const options = {
          executionProviders: [
            {
              name: provider,
              deviceType: deviceType,
              preferredLayout: provider == "webgpu" ? "NHWC" : undefined,
            },
          ],
          enableGraphCapture: provider == "webgpu",
          graphOptimizationLevel: "all",
          logSeverityLevel: 0,
        };
        inputOptions.inputLayout = "nhwc";
        inputOptions.inputShape =
          resolutionType == "general" ? [1, 256, 256, 3] : [1, 144, 256, 3];
        sess = await ort.InferenceSession.create(
          `./selfie_segmenter_${resolutionType}_19.onnx`,
          options
        );
      } else {
        // wnnModel =
        //   resolutionType == "landscape"
        //     ? new WebnnSelfieSegmenterLandscape(deviceType)
        //     : new WebnnSelfieSegmenterGeneral(deviceType);
        // const graph = await wnnModel.load({ deviceType });

        // Select model based on preferred input layout
        const context = await navigator.ml.createContext({deviceType});
        const layout = context.opSupportLimits().preferredInputLayout;
        if (layout == 'nhwc') {
          wnnModel = new SelfieSegmenterNhwc();
          wnnModel.layout = 'nhwc';
          wnnModel.inputShape = [1,144,256,3]
        } else {
          wnnModel = new SelfieSegmenter();
          wnnModel.layout = 'nhwc'
          wnnModel.inputShape = [1,144,256,3]
        }
        inputOptions.inputLayout = wnnModel.layout;
        inputOptions.inputShape = wnnModel.inputShape;
        console.log(
          `- Loading WebNN model: [${resolutionType}] deviceType: [${deviceType}] preferredLayout: [${wnnModel.layout}]`
        );
        // await wnnModel.build(graph);
        await wnnModel.build({ deviceType });
      }
      loadTime = performance.now() - start;
      console.log(`  done in ${loadTime.toFixed(2)} ms.`);
      // UI shows model building progress
      await ui.showProgressComponent("done", "current", "pending");
    }
    startRun = performance.now();
    // UI shows inferencing progress
    await ui.showProgressComponent("done", "done", "current");
    if (inputType === "image") {
      const inputBuffer = utils.getInputTensor(imgElement, inputOptions);
      console.log("- Computing... ");
      let medianComputeTime;

      console.log("- Warmup... ");
      outputBuffer = await compute(modelType, inputBuffer);
      console.log("- Warmup done... ");

      if (perfTest) {
        while (performance.now() - startRun <= 1000 * 60 * numMinutes) {
          // only record for 1 minute
          count++;
          const start = performance.now();
          outputBuffer = await compute(modelType, inputBuffer);
          const time = performance.now() - start;
          console.log(`  compute time ${count}: ${time.toFixed(2)} ms`);
          if (count > 3) {
            // skip first 3 inferences, treat as warmup
            computeTimeArray.push(time);
          }
        }
      } else {
        for (let i = 0; i < numRuns; i++) {
          const start = performance.now();
          outputBuffer = await compute(modelType, inputBuffer);
          const time = performance.now() - start;
          console.log(`  compute time ${i + 1}: ${time.toFixed(2)} ms`);
          computeTimeArray.push(time);
        }
      }

      computeTime = utils.getMedianValue(computeTimeArray);
      computeTimeArray = [];
      count = 0;
      if (numRuns > 1 || perfTest) {
        medianComputeTime = computeTime;
      }

      await ui.showProgressComponent("done", "done", "done");
      $("#fps").hide();
      ui.readyShowResultComponents();
      await drawOutput(outputBuffer, imgElement);
      showPerfResult(medianComputeTime);
    } else if (inputType === "camera") {
      count = 0;
      stream = await utils.getMediaStream({width: inputOptions.inputResolution[0], height: inputOptions.inputResolution[1]});
      camElement.srcObject = stream;
      camElement.onloadedmediadata = await renderCamStream();
      await ui.showProgressComponent("done", "done", "done");
      $("#fps").show();
      ui.readyShowResultComponents();
    } else {
      throw Error(`Unknown inputType ${inputType}`);
    }
  } catch (error) {
    console.log(error);
    ui.addAlert(error.message);
  }
  ui.handleClick(disabledSelectors, false);
}
