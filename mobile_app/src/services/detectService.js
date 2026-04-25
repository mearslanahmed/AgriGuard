import * as SecureStore from 'expo-secure-store';

const FLASK_URL = 'http://192.168.1.24:5000'; // Flask API IP

export const detectDisease = async (imageUri) => {
    // Building multipart form data - same as curl -F
    const formData = new FormData();
    formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'crop.jpg',
    });

    const response = await fetch(`${FLASK_URL}/predict`, {
        method: 'POST',
        body: formData,
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    if (!response.ok) {
        throw new Error('Prediction failed. Try again.');
    }

    return await response.json(); // {class, confidence} expected
}