const axios = require('axios');

const ESP_CAM_URL = `http://${process.env.ESP_CAM_IP}`;

const captureImage = async () => {
  const response = await axios.get(`${ESP_CAM_URL}/capture`, {
    responseType: 'arraybuffer',
    timeout: 10000,
  });
  return Buffer.from(response.data);
};

const getStatus = async () => {
  const response = await axios.get(`${ESP_CAM_URL}/status`, {
    timeout: 5000,
  });
  return response.data;
};

module.exports = { captureImage, getStatus };