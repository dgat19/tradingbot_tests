// src/components/Sidebar.js
import React from 'react';
import { List, ListItem, ListItemText } from '@mui/material';

function Sidebar() {
  const trendingTickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA'];

  return (
    <div style={{ width: '20%', marginRight: '20px' }}>
      <List component="nav" aria-label="main mailbox folders">
        {trendingTickers.map((ticker) => (
          <ListItem button key={ticker}>
            <ListItemText primary={ticker} />
          </ListItem>
        ))}
      </List>
    </div>
  );
}

export default Sidebar;
