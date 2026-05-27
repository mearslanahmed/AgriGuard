require('dotenv').config();
console.log('ESP_CAM_IP:', process.env.ESP_CAM_IP);
const express = require('express');
const cors = require('cors');
const connectDB = require('./config/db');

const app = express();

// Connect to MongoDB
connectDB();

// Middleware
app.use(cors());
app.use(express.json()); // parse JSON request bodies

// Routes
app.use('/api/auth', require('./routes/authRoutes'));
app.use('/api/pesticides', require('./routes/pesticideRoutes'));
app.use('/api/scans', require('./routes/scanRoutes'));
app.use('/api/esp', require('./routes/espRoutes')); 

// Health check — useful to confirm server is running
app.get('/health', (req, res) => {
  res.json({ status: 'AgriGuard backend running' });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ message: 'Route not found.' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});