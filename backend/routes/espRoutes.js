const express = require('express');
const router = express.Router();
const { capture, status } = require('../controllers/espController');
const { protect } = require('../middleware/authMiddleware');

router.get('/capture', protect, capture);
router.get('/status', protect, status);

module.exports = router;