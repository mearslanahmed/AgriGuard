import * as SecureStore from 'expo-secure-store';
import { FLASK_URL, BACKEND_URL } from '../config';

export const detectDisease = async (imageUri) => {
  // Step 1: Send image to Flask ML API
  const formData = new FormData();
  formData.append('image', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'crop.jpg',
  });

  const mlResponse = await fetch(`${FLASK_URL}/predict`, {
    method: 'POST',
    body: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  if (!mlResponse.ok) throw new Error('ML prediction failed. Try again.');
  const mlResult = await mlResponse.json();

  const token = await SecureStore.getItemAsync('userToken');

  // Step 2: Fetch pesticide advisory from backend
  const encodedClass = encodeURIComponent(mlResult.class_name);
  let pesticideData = null;
  try {
    const pesticideResponse = await fetch(`${BACKEND_URL}/api/pesticide/${encodedClass}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (pesticideResponse.ok) {
      pesticideData = await pesticideResponse.json();
    }
  } catch (err) {
    console.log('Pesticide fetch failed:', err.message);
  }

  // Step 3: Save scan to MongoDB for history — non-blocking
  try {
    await fetch(`${BACKEND_URL}/api/scans`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        class_name: mlResult.class_name,
        crop: mlResult.crop,
        disease: mlResult.disease,
        confidence: mlResult.confidence,
        is_healthy: mlResult.is_healthy,
        image_uri: imageUri,
      }),
    });
  } catch (err) {
    console.log('Scan save failed:', err.message);
  }

  return { mlResult, pesticideData };
};