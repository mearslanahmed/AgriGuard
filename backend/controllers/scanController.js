const Scan = require('../models/Scan');

// POST /api/scans — save a new scan result after detection
const createScan = async (req, res) => {
  try {
    const { class_name, crop, disease, confidence, is_healthy, image_uri } = req.body;

    const scan = await Scan.create({
      user: req.user._id, // injected by authMiddleware
      class_name,
      crop,
      disease,
      confidence,
      is_healthy,
      image_uri,
    });

    res.status(201).json(scan);
  } catch (err) {
    console.error('Create scan error:', err.message);
    res.status(500).json({ message: 'Failed to save scan.' });
  }
};

// GET /api/scans — fetch scans for logged in user, newest first
const getScans = async (req, res) => {
  try {
    // Optional limit param — home screen uses ?limit=3
    const limit = parseInt(req.query.limit) || 20;

    const scans = await Scan.find({ user: req.user._id })
      .sort({ createdAt: -1 })
      .limit(limit);

    res.json(scans);
  } catch (err) {
    console.error('Get scans error:', err.message);
    res.status(500).json({ message: 'Failed to fetch scans.' });
  }
};

// DELETE /api/scans/:id — farmer can delete a single scan record
const deleteScan = async (req, res) => {
  try {
    const scan = await Scan.findById(req.params.id);

    if (!scan) return res.status(404).json({ message: 'Scan not found.' });

    // Make sure the scan belongs to the requesting user
    if (scan.user.toString() !== req.user._id.toString()) {
      return res.status(403).json({ message: 'Not authorized to delete this scan.' });
    }

    await scan.deleteOne();
    res.json({ message: 'Scan deleted.' });
  } catch (err) {
    console.error('Delete scan error:', err.message);
    res.status(500).json({ message: 'Failed to delete scan.' });
  }
};

module.exports = { createScan, getScans, deleteScan };