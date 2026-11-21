import {
    pipeline,
    RawImage
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");
const status = document.getElementById("status");
const objects = document.getElementById("objects");

// Offscreen canvas to grab clean frames from the video stream for inference
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let cameraStream = null;
let detector = null;
let detecting = false;

async function loadModel() {
    try {
        status.textContent = "Loading Hugging Face AI model...";

        // Supported object-detection model for Transformers.js
        detector = await pipeline(
            "object-detection",
            "Xenova/detr-resnet-50"
        );

        status.textContent = "✅ AI model loaded! Start the camera.";
    } catch (error) {
        console.error("Failed to load model:", error);
        status.textContent = "❌ Error loading AI model. Check console (F12).";
    }
}

loadModel();

startButton.addEventListener("click", async () => {
    try {
        if (!detector) {
            status.textContent = "⏳ Model is still loading, please wait...";
            return;
        }

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        await new Promise((resolve) => {
            if (video.videoWidth > 0 && video.videoHeight > 0) {
                resolve();
            } else {
                video.onloadedmetadata = () => resolve();
            }
        });

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;

        status.textContent = "📷 Camera started — detecting objects...";
        detect();
    } catch (error) {
        console.error("Camera start error:", error);
        status.textContent = "❌ Camera error: " + error.message;
    }
});

stopButton.addEventListener("click", () => {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;
    detecting = false;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    objects.innerHTML = "No objects detected yet.";
    status.textContent = "Camera stopped.";
});

async function detect() {
    if (!cameraStream || !detector || detecting) {
        return;
    }

    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
        requestAnimationFrame(detect);
        return;
    }

    detecting = true;

    try {
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

        const image = await RawImage.fromCanvas(captureCanvas);

        const results = await detector(image, {
            threshold: 0.5
        });

        console.log("Detected results:", results);

        drawResults(results);
        showObjects(results);
    } catch (error) {
        console.error("Detection error:", error);
        status.textContent = "⚠️ Detection error: " + error.message;
    } finally {
        detecting = false;

        if (cameraStream) {
            requestAnimationFrame(detect);
        }
    }
}

function drawResults(results) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    results.forEach(result => {
        const box = result.box;
        const width = box.xmax - box.xmin;
        const height = box.ymax - box.ymin;

        ctx.strokeStyle = "#ff7900";
        ctx.lineWidth = 3;
        ctx.strokeRect(box.xmin, box.ymin, width, height);

        const label = `${result.label} ${Math.round(result.score * 100)}%`;

        ctx.fillStyle = "#ff7900";
        ctx.font = "bold 18px Arial";
        ctx.fillText(label, box.xmin, Math.max(box.ymin - 8, 20));
    });
}

function showObjects(results) {
    objects.innerHTML = "";

    if (!results || results.length === 0) {
        objects.innerHTML = "No objects detected.";
        return;
    }

    results.forEach(result => {
        const div = document.createElement("div");
        div.className = "object";
        div.innerHTML = `
            <strong>${result.label}</strong>
            <span>${Math.round(result.score * 100)}%</span>
        `;
        objects.appendChild(div);
    });
}