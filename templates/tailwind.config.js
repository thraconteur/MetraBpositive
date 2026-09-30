module.exports = {
  content: ['./templates/index.html'],
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'system-ui', 'sans-serif'] },
      colors: {
        critical: '#E53935', major: '#F57C00', medium: '#FFB300', minor: '#FDD835',
        pass: '#43A047', dark: '#1c1c1e', darkNav: '#2c2c2e',
      },
    },
  },
  plugins: [],
};
