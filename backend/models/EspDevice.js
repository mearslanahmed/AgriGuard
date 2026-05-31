const mongoose = require('mongoose');

const espDeviceSchema = new mongoose.Schema({
  deviceId: { type: String, required: true, unique: true },
  ip: { type: String, required: true },
  lastSeen: { type: Date, default: Date.now },
});

module.exports = mongoose.model('EspDevice', espDeviceSchema);