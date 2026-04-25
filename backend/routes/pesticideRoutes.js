const express = require('express');
const router = express.Router();
const { getPesticideByClass, getAllPesticides } = require('../controllers/pesticideController');
const { protect } = require('../middleware/authMiddleware');

// Protected - only logged in farmers can access
router.get('/:class_name', protect, getPesticideByClass);
router.get('/', protect, getAllPesticides);

module.exports = router;