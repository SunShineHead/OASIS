{
  "title": "GPU Cost Allocation Dashboard",
  "uid": "gpu-cost-alloc",
  "schemaVersion": 36,
  "tags": ["gpu", "cost", "dcgm", "billing"],
  "timezone": "browser",
  "refresh": "30s",
  "templating": {
    "list": [
      {
        "name": "namespace",
        "type": "query",
        "datasource": "Prometheus",
        "query": "label_values(DCGM_FI_DEV_GPU_UTIL{namespace!=\"\"}, namespace)",
        "includeAll": true,
        "refresh": 1
      }
    ]
  },
  "panels": [
    {
      "id": 1,
      "type": "stat",
      "title": "Total GPU Cost (USD/hr)",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "sum(DCGM_FI_DEV_GPU_UTIL{namespace!=\"\"} / 100 * 2)",
          "legendFormat": "Total GPU Cost"
        }
      ],
      "options": { "colorMode": "value", "graphMode": "none" }
    },
    {
      "id": 2,
      "type": "table",
      "title": "GPU Cost by Namespace (USD/hr)",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "sum by (namespace) (DCGM_FI_DEV_GPU_UTIL{namespace!=\"\"} / 100 * 2)"
        }
      ],
      "options": { "showHeader": true }
    },
    {
      "id": 3,
      "type": "table",
      "title": "GPU Cost by Pod (USD/hr)",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "sum by (pod, namespace) (DCGM_FI_DEV_GPU_UTIL{pod!=\"\"} / 100 * 2)"
        }
      ]
    },
    {
      "id": 4,
      "type": "barGauge",
      "title": "Top GPU Spenders (Namespaces)",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "sum by (namespace) (DCGM_FI_DEV_GPU_UTIL{namespace!=\"\"} / 100 * 2)"
        }
      ],
      "options": {
        "orientation": "horizontal",
        "displayMode": "gradient"
      }
    },
    {
      "id": 5,
      "type": "table",
      "title": "GPU Memory Usage Cost Weight (Normalized)",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "sum by (namespace) (DCGM_FI_DEV_FB_USED{namespace!=\"\"}) / sum(DCGM_FI_DEV_FB_TOTAL)"
        }
      ],
      "options": { "showHeader": true }
    },
    {
      "id": 6,
      "type": "graph",
      "title": "GPU Power Cost (proportional, USD/hr)",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "sum by (namespace) ((DCGM_FI_DEV_POWER_USAGE{namespace!=\"\"} / 300) * 2)"
        }
      ],
      "yaxes": [
        { "format": "currencyUSD" },
        { "format": "short" }
      ]
    },
    {
      "id": 7,
      "type": "table",
      "title": "Wasted GPU Cost (Idle Pods)",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "sum by (pod, namespace) ((DCGM_FI_DEV_GPU_UTIL{pod!=\"\"} < 5) * 2)"
        }
      ]
    },
    {
      "id": 8,
      "type": "stat",
      "title": "Idle GPU Count",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "count(DCGM_FI_DEV_GPU_UTIL == 0)"
        }
      ]
    },
    {
      "id": 9,
      "type": "heatmap",
      "title": "Cost-Weighted GPU Utilization Heatmap",
      "datasource": "Prometheus",
      "targets": [
        {
          "expr": "(DCGM_FI_DEV_GPU_UTIL{pod!=\"\"} / 100) * 2",
          "legendFormat": "{{pod}} ({{namespace}})"
        }
      ],
      "heatmap": { "colorScheme": "interpolateTurbo" }
    }
  ]
}