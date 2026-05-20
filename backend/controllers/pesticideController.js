const Pesticide = require('../models/Pesticide');

// GET /api/pesticides/:class_name
const getPesticideByClass = async (req, res) => {
  try {
    const rawParam = decodeURIComponent(req.params.class_name).trim();

    // Escape special regex characters like ( ) so they are treated as literal text
    const escapedParam = rawParam.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');

    // Anchored case-insensitive match on the escaped string literal
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