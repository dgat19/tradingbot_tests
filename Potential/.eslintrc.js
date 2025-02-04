module.exports = {
    env: {
      browser: true,
      es2021: true,
    },
    "extends": [
  "react-app",
  "plugin:react/recommended",
  "plugin:jsx-a11y/recommended",
  "plugin:import/errors",
  "plugin:import/warnings",
  "plugin:import/typescript",
  "plugin:prettier/recommended"
],
"plugins": [
  "react",
  "jsx-a11y",
  "import",
  "react-hooks",
  "prettier"
],
"rules": {
  "prettier/prettier": "error",
  // Other rules...
}
,
  };
  