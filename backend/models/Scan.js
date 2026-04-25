const mongoose = require('mongoose');

// Stores every detection result - links back to user who scanned
const scanSchema = new mongoose.Schema(
    {
        user: {
            type: mongoose.Schema.Types.ObjectId,
            ref: 'User',
            required: true,
        },
        crop: { type: String, required: true },
        disease: { type: String, required: true },
        confidence: { type: Number, required: true },
        is_healthy: { type: Boolean, default: false },
        image_uri: { type: String }, // local URI from phone — for display in history
    },
  { timestamps: true } // createdAt used for history sorting
);

module.exports = mongoose.model('Scan', scanSchema);