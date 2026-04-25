import * as SecureStore from 'expo-secure-store';

const FLASK_URL = 'http://192.168.1.24:5000'; // Flask API IP
const BACKEND_URL = 'http://192.168.1.24:3000';

export const detectDisease = async (imageUri) => {
    // Building multipart form data - same as curl -F
    // Step 1: Send image to Flask ML API
    const formData = new FormData();
    formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'crop.jpg',
    });

    const mlresponse = await fetch(`${FLASK_URL}/predict`, {
        method: 'POST',
        body: formData,
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    if (!mlresponse.ok) {
        throw new Error('ML prediction failed. Try again.');
        const mlResult = await mlResponse.json();

        // Step 2: Fetch pesticide advisory from Node backend using detected class
        const token = await SecureStore.getItemAsync('userToken');
        const encodedClass = encodeURIComponent(mlResult.class_name);

        const pesticideResponse = await fetch(`${BACKEND_URL}/api/pesticide/${encodedClass}`, {
            headers: { Authorization: `Bearer ${token}` },
        });

        // Don't crash if pesticide data missing - just retunr ML result alone
        let pesticideData = null;
        if (pesticideResponse.ok) {
            pesticideData = await pesticideResponse.json();
        }
    }

    return { mlResult, pesticideData };
};