const EspDevice = require('../models/EspDevice');
const { captureImage, getStatus, getWroomStatus, pumpOn, pumpOff, autoWater, autoOn, autoOff } = require('../services/espService');

const capture = async (req, res) => {
  try {
    const imageBuffer = await captureImage();
    res.set('Content-Type', 'image/jpeg');
    res.send(imageBuffer);
  } catch (err) {
    console.error('ESP capture error:', err.message);
    res.status(503).json({ message: 'ESP32-CAM unavailable.' });
  }
};

const status = async (req, res) => {
  try {
    const data = await getStatus();
    res.json(data);
  } catch (err) {
     console.error('ESP capture error:', err.code, err.message);
    res.status(503).json({ status: 'offline' });
  }
};


const wroomStatus = async (req, res) => {
  try {
    const data = await getWroomStatus();
    res.json(data);
  } catch (err) {
    console.error('WROOM status error:', err.code, err.message);
    res.status(503).json({ status: 'offline', message: 'ESP32-WROOM unavailable.' });
  }
};

const pumpOnHandler = async (req, res) => {
  try {
    const data = await pumpOn();
    res.json(data);
  } catch (err) {
    console.error('WROOM pump on error:', err.code, err.message);
    res.status(503).json({ message: 'ESP32-WROOM unavailable.' });
  }
};

const pumpOffHandler = async (req, res) => {
  try {
    const data = await pumpOff();
    res.json(data);
  } catch (err) {
    console.error('WROOM pump off error:', err.code, err.message);
    res.status(503).json({ message: 'ESP32-WROOM unavailable.' });
  }
};

const autoWaterHandler = async (req, res) => {
  try {
    const data = await autoWater();
    res.json(data);
  } catch (err) {
    console.error('WROOM auto error:', err.code, err.message);
    res.status(503).json({ message: 'ESP32-WROOM unavailable.' });
  }
};

const autoOnHandler = async (req, res) => {
  try {
    const data = await autoOn();
    res.json(data);
  } catch (err) {
    console.error('WROOM auto on error:', err.code, err.message);
    res.status(503).json({ message: 'ESP32-WROOM unavailable.' });
  }
};

const autoOffHandler = async (req, res) => {
  try {
    const data = await autoOff();
    res.json(data);
  } catch (err) {
    console.error('WROOM auto off error:', err.code, err.message);
    res.status(503).json({ message: 'ESP32-WROOM unavailable.' });
  }
};

const registerDevice = async (req, res) => {
  try {
    const { deviceId, ip } = req.body;
    await EspDevice.findOneAndUpdate(
      { deviceId },
      { ip, lastSeen: Date.now() },
      { upsert: true, new: true }
    );
    res.json({ message: 'Device registered.', deviceId, ip });
  } catch (err) {
    console.error('Device register error:', err.message);
    res.status(500).json({ message: 'Registration failed.' });
  }
};

const listDevices = async (req, res) => {
  try {
    const devices = await EspDevice.find({});
    res.json(devices);
  } catch (err) {
    console.error('List devices error:', err.message);
    res.status(500).json({ message: 'Failed to retrieve devices.' });
  }
};

module.exports = { capture, status, wroomStatus, pumpOnHandler, pumpOffHandler, autoWaterHandler, autoOnHandler, autoOffHandler, registerDevice, listDevices };