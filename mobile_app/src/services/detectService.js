import * as SecureStore from 'expo-secure-store';
import { FLASK_URL, BACKEND_URL } from '../config';
import * as ImageManipulator from 'expo-image-manipulator';

export const detectDisease = async (imageUri) => {
  // Compressing and resizing the image BEFORE sending
  const compressesImage = await ImageManipulator.manipulateAsync(
    imageUri,
    [{ resize: { width: 800 } }], // Resize to width of 800px, maintaining aspect ratio
    { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG } // Compress to 70% quality
  );
  // Step 1: Send image to Flask ML API
  const formData = new FormData();
  formData.append('image', {
    uri: compressesImage.uri,   // Use the compressed image URI
    type: 'image/jpeg',
    name: 'crop.jpg',
  });

  const mlResponse = await fetch(`${FLASK_URL}/predict`, {
  method: 'POST',
  body: formData,
  headers: { 'Content-Type': 'multipart/form-data' },
});

const mlResult = await mlResponse.json();

// 422 = not a plant image
if (mlResponse.status === 422) {
  throw new Error(mlResult.error);
}

if (!mlResponse.ok) {
  throw new Error('ML prediction failed. Try again.');
}

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