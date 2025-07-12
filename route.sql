SELECT
  DATE_FORMAT(datedriven, "%Y-%m-%d") as date,
  SUM(distance / 1000) AS distance_km,
  COUNT(
    DISTINCT(rego)
  ) AS utes_driven,
  DAYOFWEEK(datedriven) AS day_of_week,
  CASE WHEN DAYOFWEEK(datedriven) IN (1, 7) THEN 1 ELSE 0 end AS is_weekend,
  MONTH(datedriven) AS month,
  round(
    0.5 - 0.5 * Sin(
      2 * Pi() * (
        DAYOFYEAR(datedriven) -14
      ) / 365.0
    ),
    4
  ) AS seasonal_scalar,
  DATEDIFF(datedriven, '2023-11-11') AS days_in_business
FROM
  routes
WHERE
  paid = 1
GROUP BY
  date
ORDER BY date
