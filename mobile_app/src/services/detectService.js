import * as SecureStore from 'expo-secure-store';
import { FLASK_URL, BACKEND_URL } from '../config';
import * as ImageManipulator from 'expo-image-manipulator';

export const detectDisease = async (imageUri) => {
  try {
    // Compressing and resizing the image BEFORE sending to avoid OOM crashes
    const compressesImage = await ImageManipulator.manipulateAsync(
      imageUri,
      [{ resize: { width: 800 } }], 
      { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG } 
    );

    // Step 1: Send image to Flask ML API
    const formData = new FormData();
    formData.append('image', {
      uri: compressesImage.uri,   
      type: 'image/jpeg',
      name: 'crop.jpg',
    });

    const mlResponse = await fetch(`${FLASK_URL}/predict`, {
      method: 'POST',
      body: formData,
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    const mlResult = await mlResponse.json();

    // 422 = Unrecognized / Below custom thresholds
    if (mlResponse.status === 422) {
      throw new Error(mlResult.error);
    }

    if (!mlResponse.ok) {
      throw new Error('ML prediction failed. Try again.');
    }

    const token = await SecureStore.getItemAsync('userToken');

    // Step 2: Fetch pesticide advisory from backend using verified plural route and keys
    const encodedClass = encodeURIComponent(mlResult.disease_label.trim());
    let pesticideData = null;
    
    try {
      const pesticideResponse = await fetch(`${BACKEND_URL}/api/pesticides/${encodedClass}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (pesticideResponse.ok) {
        pesticideData = await pesticideResponse.json();
      } else {
        console.log(`[MOBILE FETCH 404/500] Backend status: ${pesticideResponse.status}`);
      }
    } catch (err) {
      console.log('Pesticide fetch failed:', err.message);
    }

    // Step 3: Save scan to MongoDB for history — non-blocking pass
    try {
      await fetch(`${BACKEND_URL}/api/scans`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          class_name: mlResult.disease_label, // Saved using correct field pairing
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

  } catch (globalError) {
    console.error('Core Detection Failed:', globalError.message);
    throw globalError;
  }
};