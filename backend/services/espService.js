const axios = require('axios');
const EspDevice = require('../models/EspDevice');

// Fetch device IP from DB by deviceId — avoids hardcoded IPs
const getDeviceIP = async (deviceId) => {
  const device = await EspDevice.findOne({ deviceId });
  if (!device) throw new Error(`Device ${deviceId} not registered.`);
  return `http://${device.ip}`;
};

const captureImage = async () => {
  const url = await getDeviceIP('esp-cam');
  const response = await axios.get(`${url}/capture`, {
    responseType: 'arraybuffer', timeout: 10000,
  });
  return Buffer.from(response.data);
};

const getStatus = async () => {
  const url = await getDeviceIP('esp-cam');
  const response = await axios.get(`${url}/status`, { timeout: 5000 });
  return response.data;
};

const getWroomStatus = async () => {
  const url = await getDeviceIP('esp-wroom');
  const response = await axios.get(`${url}/status`, { timeout: 5000 });
  return response.data;
};

const pumpOn = async () => {
  const url = await getDeviceIP('esp-wroom');
  const response = await axios.get(`${url}/pump/on`, { timeout: 5000 });
  return response.data;
};

const pumpOff = async () => {
  const url = await getDeviceIP('esp-wroom');
  const response = await axios.get(`${url}/pump/off`, { timeout: 5000 });
  return response.data;
};

const autoWater = async () => {
  const url = await getDeviceIP('esp-wroom');
  const response = await axios.get(`${url}/auto`, { timeout: 5000 });
  return response.data;
};

const autoOn = async () => {
  const url = await getDeviceIP('esp-wroom');
  const response = await axios.get(`${url}/auto/on`, { timeout: 5000 });
  return response.data;
};

const autoOff = async () => {
  const url = await getDeviceIP('esp-wroom');
  const response = await axios.get(`${url}/auto/off`, { timeout: 5000 });
  return response.data;
};

module.exports = { captureImage, getStatus, getWroomStatus, pumpOn, pumpOff, autoWater, autoOn, autoOff };