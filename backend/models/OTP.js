const mongoose = require('mongoose');

const otpSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    lowercase: true,
    trim: true
  },
  otp: {
    type: String,
    required: true
  },
  purpose: {
    type: String,
    enum: ['register', 'reset'],
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now,
    expires: 300 // Document self-destructs after 5 minutes
  }
});

module.exports = mongoose.model('OTP', otpSchema);