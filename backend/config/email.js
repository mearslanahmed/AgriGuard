const nodemailer = require('nodemailer');

const transporter = nodemailer.createTransport({
  service: 'gmail',
  host: 'smtp.gmail.com',
  port: 465,
  secure: true, 
  auth: {
    user: process.env.EMAIL_USER,     
    pass: process.env.EMAIL_APP_PASS  
  },
  pool: true 
});

transporter.verify((error, success) => {
  if (error) {
    console.error('Mail Carrier Relay Refused Handshake:', error.message);
  } else {
    console.log('AgriGuard Secure Mail Carrier Initialized and Active');
  }
});

module.exports = transporter;