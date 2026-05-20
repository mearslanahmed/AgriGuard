require('dotenv').config({ path: require('path').resolve(__dirname, '../.env') });
const mongoose = require('mongoose');
const Pesticide = require('../models/Pesticide');

const data = [
  // ─── TOMATO ───────────────────────────────────────────────
  {
    class_name: 'Tomato__Tomato_mosaic_virus',
    crop: 'Tomato',
    disease: 'Tomato Mosaic Virus',
    is_healthy: false,
    pesticide: {
      name: 'No chemical cure available',
      dosage: 'N/A',
      spray_interval: 'N/A',
      water_ratio: 'N/A',
      safety: 'Remove and destroy infected plants immediately',
      notes: 'Control aphids with Imidacloprid 0.5ml/L to prevent spread. Disinfect tools after use.',
    },
  },
  {
    class_name: 'Tomato__Target_Spot',
    crop: 'Tomato',
    disease: 'Target Spot',
    is_healthy: false,
    pesticide: {
      name: 'Chlorothalonil',
      dosage: '2g per litre of water',
      spray_interval: 'Every 7 days',
      water_ratio: '2g / 1L',
      safety: 'Wear gloves and mask. Avoid spraying near water sources.',
      notes: 'Apply early morning or evening. Improve air circulation around plants.',
    },
  },
  {
    class_name: 'Tomato_Yellow_Leaf_Curl_Virus',
    crop: 'Tomato',
    disease: 'Yellow Leaf Curl Virus',
    is_healthy: false,
    pesticide: {
      name: 'Imidacloprid',
      dosage: '0.5ml per litre of water',
      spray_interval: 'Every 10 days',
      water_ratio: '0.5ml / 1L',
      safety: 'Avoid contact with skin. Wash hands thoroughly after use.',
      notes: 'Targets whiteflies that spread the virus. Remove infected plants to limit spread.',
    },
  },
  {
    class_name: 'Tomato_Leaf_Mold',
    crop: 'Tomato',
    disease: 'Leaf Mold',
    is_healthy: false,
    pesticide: {
      name: 'Mancozeb',
      dosage: '2.5g per litre of water',
      spray_interval: 'Every 7-10 days',
      water_ratio: '2.5g / 1L',
      safety: 'Use protective clothing. Do not inhale spray mist.',
      notes: 'Ensure good ventilation in greenhouse conditions. Avoid overhead watering.',
    },
  },
  {
    class_name: 'Tomato_Late_blight',
    crop: 'Tomato',
    disease: 'Late Blight',
    is_healthy: false,
    pesticide: {
      name: 'Metalaxyl + Mancozeb',
      dosage: '2g per litre of water',
      spray_interval: 'Every 7 days',
      water_ratio: '2g / 1L',
      safety: 'Wear full protective gear. Keep children away during application.',
      notes: 'Most effective when applied preventively. Destroy infected plant debris after harvest.',
    },
  },
  {
    class_name: 'Tomato_Early_blight',
    crop: 'Tomato',
    disease: 'Early Blight',
    is_healthy: false,
    pesticide: {
      name: 'Copper Oxychloride',
      dosage: '2g per litre of water',
      spray_interval: 'Every 7 days',
      water_ratio: '2g / 1L',
      safety: 'Wear gloves and mask during application.',
      notes: 'Remove infected lower leaves before spraying. Rotate crops next season.',
    },
  },
  {
    class_name: 'Tomato_Spider_mites Two-spotted_spider_mite',
    crop: 'Tomato',
    disease: 'Spider Mites (Two-spotted)',
    is_healthy: false,
    pesticide: {
      name: 'Abamectin',
      dosage: '1ml per litre of water',
      spray_interval: 'Every 5-7 days',
      water_ratio: '1ml / 1L',
      safety: 'Highly toxic — wear full protective gear. Keep away from children.',
      notes: 'Spray undersides of leaves where mites cluster. Avoid during high temperatures.',
    },
  },
  {
    class_name: 'Tomato_Septoria_leaf_spot',
    crop: 'Tomato',
    disease: 'Septoria Leaf Spot',
    is_healthy: false,
    pesticide: {
      name: 'Chlorothalonil',
      dosage: '2g per litre of water',
      spray_interval: 'Every 7-10 days',
      water_ratio: '2g / 1L',
      safety: 'Avoid skin and eye contact. Use in well-ventilated area.',
      notes: 'Remove affected leaves before treatment. Avoid wetting foliage when watering.',
    },
  },
  {
    class_name: 'Tomato_bacterial_spot',
    crop: 'Tomato',
    disease: 'Bacterial Spot',
    is_healthy: false,
    pesticide: {
      name: 'Copper Hydroxide',
      dosage: '2g per litre of water',
      spray_interval: 'Every 7 days',
      water_ratio: '2g / 1L',
      safety: 'Wear protective gloves. Wash equipment thoroughly after use.',
      notes: 'Apply before rain if possible. Avoid working with wet plants to prevent spread.',
    },
  },
  {
    class_name: 'Tomato_healthy',
    crop: 'Tomato',
    disease: 'Healthy',
    is_healthy: true,
    pesticide: null,
  },

  // ─── POTATO ───────────────────────────────────────────────
  {
    class_name: 'Potato___Late_blight',
    crop: 'Potato',
    disease: 'Late Blight',
    is_healthy: false,
    pesticide: {
      name: 'Metalaxyl + Mancozeb',
      dosage: '2.5g per litre of water',
      spray_interval: 'Every 7 days',
      water_ratio: '2.5g / 1L',
      safety: 'Wear mask and gloves. Do not eat or drink during application.',
      notes: 'Most destructive potato disease — act fast. Hill up soil around plants to protect tubers.',
    },
  },
  {
    class_name: 'Potato___Early_blight',
    crop: 'Potato',
    disease: 'Early Blight',
    is_healthy: false,
    pesticide: {
      name: 'Mancozeb',
      dosage: '2g per litre of water',
      spray_interval: 'Every 10 days',
      water_ratio: '2g / 1L',
      safety: 'Avoid inhaling spray. Use gloves.',
      notes: 'Start spraying when plants are 15cm tall as a preventive measure.',
    },
  },
  {
    class_name: 'Potato___healthy',
    crop: 'Potato',
    disease: 'Healthy',
    is_healthy: true,
    pesticide: null,
  },

  // ─── PEPPER ───────────────────────────────────────────────
  {
    class_name: 'Pepper__bell___Bacterial_spot',
    crop: 'Pepper Bell',
    disease: 'Bacterial Spot',
    is_healthy: false,
    pesticide: {
      name: 'Copper Oxychloride',
      dosage: '2g per litre of water',
      spray_interval: 'Every 7 days',
      water_ratio: '2g / 1L',
      safety: 'Wear gloves. Avoid spraying in windy conditions.',
      notes: 'Use disease-free seeds. Avoid overhead irrigation to reduce leaf wetness.',
    },
  },
  {
    class_name: 'Pepper__bell___healthy',
    crop: 'Pepper Bell',
    disease: 'Healthy',
    is_healthy: true,
    pesticide: null,
  },

  // ─── TOMATO (background class) ────────────────────────────
  {
    class_name: 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    crop: 'Tomato',
    disease: 'Yellow Leaf Curl Virus (Variant)',
    is_healthy: false,
    pesticide: {
      name: 'Imidacloprid',
      dosage: '0.5ml per litre of water',
      spray_interval: 'Every 10 days',
      water_ratio: '0.5ml / 1L',
      safety: 'Avoid contact with skin. Wash hands thoroughly after use.',
      notes: 'Same as Yellow Leaf Curl Virus — targets whitefly vectors.',
    },
  },
];

const seed = async () => {
    try {
        await mongoose.connect(process.env.MONGO_URI);
        console.log('MongoDB connected');

        // Clear existing data first to avoid duplicates
        await Pesticide.deleteMany({});
        console.log('Cleared existing pesticide data');

        await Pesticide.insertMany(data);
        console.log(`Seeded ${data.length} pesticide records`);

        mongoose.disconnect();
    } catch (err) {
        console.error('Seeding failed', err.message);
        process.exit(1);
    }
};

seed();