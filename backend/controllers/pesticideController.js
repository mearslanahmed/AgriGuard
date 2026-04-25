const Pesticide = require('../models/Pesticide');

// GET /api/pesticides/:class_name
// Called by mobile app after detection to get advisory for detected disease

const getPesticideByClass = async (req, res) => {
    try {
        // class_name comes URL-encoded from mobile app, decode it first
        const class_name = decodeURIComponent(req.params.class_name);

        const record = await Pesticide.findOne({ class_name });

        if (!record) {
            return res.status(404).json({
                message: `No pesticide info found for class: ${class_name},`
            });
        }

        res.json(record);
    } catch (err) {
        console.error('Pesticide fetch error:', err.message);
        res.status(500).json({ message: 'Server error fetching pesticide info' });
    }
};

// GET /api/pesticides
// Admin use - return all records

const getAllPesticides = async (req, res) => {
    try {
        const records = await Pesticide.find({}).sort({ crop: 1});
        res.json(records);
    } catch (err) {
        console.error('Pesticide list error:', err.message);
        res.status(500).json({ message: 'Server error fetching pesticide list' });
    }
};

module.exports = {getPesticideByClass, getAllPesticides};