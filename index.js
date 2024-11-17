const webcamEl = document.querySelector("#webcam");
const liveView = document.querySelector("#liveView");
const appSection = document.querySelector("#app");
const btnEnableWebcam = document.querySelector("#btnEnableWebcam");
const outputMessageEl = document.querySelector("#outputMessage");

let webcam;
let model;
let knifePosition = null;
let knifeDetectionTime = null;
let knifeReminderDisplayed = false; 
const confidenceThreshold = 0.5;
const knifeReminderThreshold = 60000;

const positionTolerance = 50;

const kitchenObjects = [
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'dining table',
  'person',
];

function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js no está cargado");
  }
}

function getUserMediaSupported() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function enableCam(event) {
  if (!model) return;

  event.target.classList.add("removed");
  try {
    webcam = await tf.data.webcam(webcamEl);
    outputMessageEl.innerText = "¡Webcam habilitada! Detectando objetos...";
    await predictWebcam();
  } catch (error) {
    console.error("Acceso a la webcam denegado:", error);
    outputMessageEl.innerText = "Acceso a la webcam denegado.";
  }
}

async function loadCocoSsdModel() {
  model = await cocoSsd.load();
  btnEnableWebcam.disabled = false;
  outputMessageEl.innerText = "¡Modelo cargado! Habilita la webcam para comenzar la detección.";
}

async function predictWebcam() {
  const objects = [];

  while (true) {
    const frame = await webcam.capture();
    const predictions = await model.detect(frame);

    objects.forEach((object) => liveView.removeChild(object));
    objects.length = 0;

    let foundBottle = false;
    let foundCup = false;
    let foundPerson = false;
    let foundKnife = false;
    let currentKnifePosition = null;

    for (let n = 0; n < predictions.length; n++) {
      if (
        predictions[n].score > confidenceThreshold &&
        kitchenObjects.includes(predictions[n].class)
      ) {
        const c = predictions[n].class;
        const score = Math.round(parseFloat(predictions[n].score) * 100);

        if (c === 'bottle') foundBottle = true;
        if (c === 'cup') foundCup = true;
        if (c === 'person') foundPerson = true;
        if (c === 'knife') {
          foundKnife = true;
          currentKnifePosition = predictions[n].bbox;
        }

        const p = document.createElement("p");
        p.innerText = `${c} - con ${score}% de confianza.`;
        p.style = `margin-left: ${predictions[n].bbox[0]}px; 
                   margin-top: ${predictions[n].bbox[1] - 10}px; 
                   width: ${predictions[n].bbox[2] - 10}px; 
                   top: 0; 
                   left: 0;`;

        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");
        highlighter.style = `left: ${predictions[n].bbox[0]}px; 
                             top: ${predictions[n].bbox[1]}px; 
                             width: ${predictions[n].bbox[2]}px; 
                             height: ${predictions[n].bbox[3]}px;`;

        liveView.appendChild(highlighter);
        liveView.appendChild(p);
        objects.push(highlighter);
        objects.push(p);
      }
    }

    if (foundKnife && currentKnifePosition) {
      if (knifePosition) {
        if (arePositionsEqual(knifePosition, currentKnifePosition)) {
          if (!knifeDetectionTime) {
            knifeDetectionTime = Date.now();
            console.log("Posición del cuchillo estable, temporizador iniciado.");
          } else {
            const elapsed = Date.now() - knifeDetectionTime;
            console.log(`El cuchillo ha estado estacionario por ${elapsed / 1000} segundos.`);
            if (elapsed > knifeReminderThreshold) {
              displayKnifeReminder();
              knifeDetectionTime = null;
            }
          }
        } else {
          knifePosition = currentKnifePosition;
          knifeDetectionTime = null;
          knifeReminderDisplayed = false;
          console.log("Posición del cuchillo cambió, temporizador reiniciado.");
        }
      } else {
        knifePosition = currentKnifePosition;
        knifeDetectionTime = null;
        console.log("Cuchillo detectado, posición registrada.");
      }
    } else {
      if (knifePosition || knifeDetectionTime) {
        console.log("Cuchillo ya no detectado, posición y temporizador reiniciados.");
      }
      knifePosition = null;
      knifeDetectionTime = null;
      knifeReminderDisplayed = false;
    }

    if (knifeReminderDisplayed) {
      outputMessageEl.innerText = "Recuerda guardar el cuchillo después de usarlo.";
    } else if (foundBottle && foundCup) {
      outputMessageEl.innerText = "¿Te gustaría tomar algo?";
    } else if (foundPerson && foundKnife) {
      outputMessageEl.innerText = "Precaución: asegúrate de manejar el cuchillo de forma segura.";
    } else {
      outputMessageEl.innerText = "Detectando objetos...";
    }

    frame.dispose();
    await tf.nextFrame();
  }
}

function arePositionsEqual(pos1, pos2) {
  const [x1, y1, w1, h1] = pos1;
  const [x2, y2, w2, h2] = pos2;
  return (
    Math.abs(x1 - x2) <= positionTolerance &&
    Math.abs(y1 - y2) <= positionTolerance &&
    Math.abs(w1 - w2) <= positionTolerance &&
    Math.abs(h1 - h2) <= positionTolerance
  );
}

function displayKnifeReminder() {
  knifeReminderDisplayed = true;
  outputMessageEl.innerText = "Recuerda guardar el cuchillo después de usarlo.";
  console.log("Recordatorio mostrado: Recuerda guardar el cuchillo después de usarlo.");
}

async function app() {
  if (!getUserMediaSupported()) {
    console.warn("getUserMedia() no es soportado por tu navegador");
    outputMessageEl.innerText = "La webcam no es soportada por tu navegador.";
    return;
  }
  btnEnableWebcam.addEventListener("click", enableCam);

  await loadCocoSsdModel();
}

(async function initApp() {
  try {
    initTFJS();
    await app();
  } catch (error) {
    console.error(error);
    outputMessageEl.innerText = error.message;
  }
})();
