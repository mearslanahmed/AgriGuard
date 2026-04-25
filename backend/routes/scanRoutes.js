const express = require('express')
const router = express.Router();
const { createScan, getScans, deleteScan } = require('../controllers/scanController');
const { protect } = require('../middleware/authMiddleware');

// All scan routes require login
router.post('/', protect, createScan);
router.get('/', protect, getScans);
router.delete('/:id', protect, deleteScan);

module.exports = router;