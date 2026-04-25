import * as SecureStore from "expo-secure-store";

const FLASK_URL = 'http://192.168.1.11:5000';
const BACKEND_URL = 'http://192.168.1.11:3000';

export const detectDisease = async (imageUri) => {
  // Step 1: Send image to Flask ML API for disease prediction
  const formData = new FormData();
  formData.append("image", {
    uri: imageUri,
    type: "image/jpeg",
    name: "crop.jpg",
  });

  const mlResponse = await fetch(`${FLASK_URL}/predict`, {
    method: "POST",
    body: formData,
    headers: { "Content-Type": "multipart/form-data" },
  });

  if (!mlResponse.ok) throw new Error("ML prediction failed. Try again.");
  const mlResult = await mlResponse.json();

  // Step 2: Fetch pesticide advisory from backend using the detected class name
  const token = await SecureStore.getItemAsync("userToken");
  const encodedClass = encodeURIComponent(mlResult.class_name);

  let pesticideData = null;
  try {
    const pesticideResponse = await fetch(
      `${BACKEND_URL}/api/pesticide/${encodedClass}`,
      {
        headers: { Authorization: `Bearer ${token}` },
      },
    );
    if (pesticideResponse.ok) {
      pesticideData = await pesticideResponse.json();
    }
  } catch (err) {
    // Non-blocking — app still works without pesticide data
    console.log("Pesticide fetch failed:", err.message);
  }

  return { mlResult, pesticideData };
};