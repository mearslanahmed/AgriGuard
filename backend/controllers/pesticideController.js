const Pesticide = require('../models/Pesticide');

// GET /api/pesticides/:class_name
// Called by mobile app after detection to get advisory for detected disease
const getPesticideByClass = async (req, res) => {
  try {
    const rawParam = decodeURIComponent(req.params.class_name).trim();

    // Escape regex structural characters like ( ) so they evaluate as literal text primitives
    const escapedParam = rawParam.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');

    // Anchored case-insensitive regular expression match to eliminate string discrepancies
    const record = await Pesticide.findOne({
      disease_label: { $regex: new RegExp(`^${escapedParam}$`, 'i') }
    });

    if (!record) {
      console.log(`[DATABASE 404] No pesticide record matched the key: "${rawParam}"`);
      return res.status(404).json({
        message: `No pesticide info found for: ${rawParam}`
      });
    }

    res.json(record);
  } catch (err) {
    console.error('Pesticide fetch error:', err.message);
    res.status(500).json({ message: 'Server error fetching pesticide info' });
  }
};

const getAllPesticides = async (req, res) => {
  try {
    const records = await Pesticide.find({}).sort({ crop: 1 });
    res.json(records);
  } catch (err) {
    console.error('Pesticide list error:', err.message);
    res.status(500).json({ message: 'Server error fetching pesticide list' });
  }
};

module.exports = { getPesticideByClass, getAllPesticides };