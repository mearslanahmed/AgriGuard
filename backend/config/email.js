const nodemailer = require('nodemailer');

const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_APP_PASS?.replace(/\s/g, '').trim()
  }
});

transporter.verify((error, success) => {
  if (error) {
    console.error('Mail Carrier Relay Refused Handshake:', error.message);
  } else {
    console.log('AgriGuard Secure Mail Carrier Initialized and Active');
  }
});

module.exports = transporter;