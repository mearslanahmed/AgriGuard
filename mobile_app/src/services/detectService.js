import * as SecureStore from 'expo-secure-store';
import * as FileSystem from 'expo-file-system/legacy';
import { FLASK_URL, BACKEND_URL } from '../config';
import * as ImageManipulator from 'expo-image-manipulator';

export const detectDisease = async (imageUri) => {
  try {
    // Compress and resize before sending — prevents OOM crashes on large images
    const compressedImage = await ImageManipulator.manipulateAsync(
      imageUri,
      [{ resize: { width: 800 } }],
      { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG }
    );

    const formData = new FormData();
    formData.append('image', {
      uri: compressedImage.uri,
      type: 'image/jpeg',
      name: 'crop.jpg',
    });

    const mlResponse = await fetch(`${FLASK_URL}/predict`, {
      method: 'POST',
      body: formData,
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    const mlResult = await mlResponse.json();

    // 422 means Flask rejected the image — unrecognized crop or below confidence threshold
    if (mlResponse.status === 422) {
      throw new Error(mlResult.error);
    }

    if (!mlResponse.ok) {
      throw new Error('ML prediction failed. Try again.');
    }

    const token = await SecureStore.getItemAsync('userToken');

    // URI-encode the disease label since it contains spaces and parentheses
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

    // Save scan non-blocking — history works even if this fails
    try {
      await fetch(`${BACKEND_URL}/api/scans`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          class_name: mlResult.disease_label,
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
    // console.log instead of console.error — suppresses Expo's unstyled system toast
    console.log('Core Detection Intercept Log:', globalError.message);
    throw globalError;
  }
};

export const detectDiseaseFromESP = async () => {
  try {
    const token = await SecureStore.getItemAsync('userToken');

    // Download ESP32-CAM image to a temp file so we can pass a local URI
    // into detectDisease — which expects a file URI, not a remote URL
    const tempUri = FileSystem.cacheDirectory + 'esp_capture.jpg';
    const downloadResult = await FileSystem.downloadAsync(
      `${BACKEND_URL}/api/esp/capture`,
      tempUri,
      { headers: { Authorization: `Bearer ${token}` } }
    );

    if (downloadResult.status !== 200) {
      throw new Error('Failed to capture image from ESP32-CAM');
    }

    return await detectDisease(downloadResult.uri);

  } catch (globalError) {
    console.log('ESP Detection Error:', globalError.message);
    throw globalError;
  }
};