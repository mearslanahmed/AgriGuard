const express = require('express');
const router = express.Router();
const { capture, status, wroomStatus, pumpOnHandler, pumpOffHandler, autoWaterHandler, autoOnHandler, autoOffHandler, registerDevice } = require('../controllers/espController');
const { protect } = require('../middleware/authMiddleware');

// ESP32-CAM routes
router.get('/capture', protect, capture);
router.get('/status', protect, status);

// ESP32-WROOM routes
router.get('/wroom/status', protect, wroomStatus);
router.get('/wroom/pump/on', protect, pumpOnHandler);
router.get('/wroom/pump/off', protect, pumpOffHandler);
router.get('/wroom/auto', protect, autoWaterHandler);
router.get('/wroom/auto/on', protect, autoOnHandler);
router.get('/wroom/auto/off', protect, autoOffHandler);

router.post('/register', registerDevice);
router.get('/list', listDevices);

module.exports = router;