async function predict() {
    const city = document.getElementById("city").value;

    document.getElementById("result").innerHTML = "🔄 Loading...";

    const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({city})
    });

    const data = await res.json();

    let emoji = data.prediction === "Rain" ? "🌧️" : "☀️";

    document.getElementById("result").innerHTML = `
        ${emoji} <b>${data.prediction}</b><br>
        Confidence: ${data.confidence}<br>
        Mode: ${data.mode}<br>
        🌡️ ${data.temp}°C | 💧 ${data.humidity}% | 🌪️ ${data.wind}
    `;
}
