const mongoose = require('mongoose')

// Each document = one disease and its recommended persticide treatment
const pesticideSchema = new mongoose.Schema(
    {
        // Must match class_name returned by Flask API exactly
        class_name: {
            type: String,
            required: true,
            unique: true,
            trim: true,
        },
        crop: {
            type: String,
            required: true,
            trim: true,
        },
        disease: {
            type: String,
            required: true,
            trim: true,
        },
        is_healthy: {
            type: Boolean,
            default: false,
        },

        // Pesticide recommendation - null if plant is healthy
        pesticide: {
            name: String,           // e.g. "Copper Oxychloride"
            dosage: String,         // e.g. "2g per litre of water"
            spray_interval: String, // e.g. "Every 7 days"
            water_ratio: String,    // e.g. "2g / 1L"
            safety: String,         // e.g. "Wear gloves and mask during application"
            notes: String,          // any extra advice for the farmer
        },
    },
    { timestamps: true}
);

module.exports = mongoose.model('Pesticide', pesticideSchema);