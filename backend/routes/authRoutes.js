const express = require('express');
const router = express.Router();
const { 
  sendVerificationOTP, register, login, getMe, updateProfile, changePassword, resetPasswordWithOTP 
} = require('../controllers/authController');
const { protect } = require('../middleware/authMiddleware');

router.post('/send-otp', sendVerificationOTP);
router.post('/register', register);
router.post('/login', login);
router.post('/reset-password', resetPasswordWithOTP);

router.get('/me', protect, getMe); 
router.put('/update', protect, updateProfile);
router.put('/change-password', protect, changePassword);

module.exports = router;