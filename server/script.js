const canvas = document.getElementById("pad");
const ctx = canvas.getContext("2d");
ctx.strokeStyle = "white";
ctx.lineWidth = 15;
let drawing = false;

canvas.onmousedown = e => {
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
};

canvas.onmousemove = e => {
  if (drawing) {
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  }
};

canvas.onmouseup = () => {
  drawing = false;
  predict();
};

function clearCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").innerText = "Prediction: -";
}

async function predict() {
  const dataUrl = canvas.toDataURL("image/png");
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl })
  });
  const data = await res.json();
  document.getElementById("result").innerText = "Prediction: " + data.prediction;
}