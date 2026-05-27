const { captureImage, getStatus } = require('../services/espService');

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

module.exports = { capture, status };