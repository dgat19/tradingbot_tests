// src/components/TradingChart.js
import React, { useEffect, useRef } from 'react';

function TradingChart({ symbol = 'AAPL' }) {
  const chartContainerRef = useRef(null);
  const tvWidgetRef = useRef(null);

  useEffect(() => {
    // Clean up the widget when the component is unmounted or symbol changes
    return () => {
      if (tvWidgetRef.current) {
        tvWidgetRef.current.remove();
        tvWidgetRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const widgetOptions = {
      symbol: symbol,
      datafeed: new window.Datafeeds.UDFCompatibleDatafeed('https://YOUR_DATA_FEED_URL'),
      interval: 'D',
      container: chartContainerRef.current,
      library_path: '/charting_library/',
      locale: 'en',
      disabled_features: ['use_localstorage_for_settings'],
      enabled_features: ['study_templates'],
      charts_storage_url: 'https://saveload.tradingview.com',
      charts_storage_api_version: '1.1',
      client_id: 'your_company_name',
      user_id: 'public_user_id',
      fullscreen: false,
      autosize: true,
      theme: 'Light',
    };

    const tvWidget = new window.TradingView.widget(widgetOptions);
    tvWidgetRef.current = tvWidget;

    return () => {
      if (tvWidgetRef.current) {
        tvWidgetRef.current.remove();
        tvWidgetRef.current = null;
      }
    };
  }, [symbol]);

  return (
    <div
      ref={chartContainerRef}
      className="TVChartContainer"
      style={{ height: '600px', width: '100%' }}
    />
  );
}

export default TradingChart;
