import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const predict = async (symbol) => {
  try {
    const response = await axios.get(`${API_URL}/predict`, {
      params: { symbol },
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching prediction:', error);
    throw error;
  }
};
