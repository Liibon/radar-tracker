#include "tracker/csv.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace tracker {
namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') { field += '"'; ++i; }
      else quoted = !quoted;
    } else if (ch == ',' && !quoted) {
      fields.push_back(field); field.clear();
    } else field += ch;
  }
  fields.push_back(field);
  return fields;
}

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

}  // namespace

std::vector<Measurement> read_measurements_csv(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open measurements CSV: " + path);
  std::string line;
  if (!std::getline(input, line)) return {};
  const auto headers = split_csv_line(line);
  std::unordered_map<std::string, std::size_t> column;
  for (std::size_t i = 0; i < headers.size(); ++i) column[lowercase(headers[i])] = i;
  const char* required[] = {"timestamp", "sensor_id", "measurement_id", "type",
                            "value1", "value2", "true_target_id", "sensor_x", "sensor_y"};
  for (const char* name : required)
    if (!column.count(name)) throw std::runtime_error(std::string("missing CSV column: ") + name);

  auto optional = [&](const std::vector<std::string>& fields, const char* name,
                      double fallback) {
    const auto found = column.find(name);
    if (found == column.end() || found->second >= fields.size() || fields[found->second].empty())
      return fallback;
    return std::stod(fields[found->second]);
  };

  std::vector<Measurement> measurements;
  std::size_t row = 1;
  while (std::getline(input, line)) {
    ++row;
    if (line.empty()) continue;
    const auto fields = split_csv_line(line);
    auto get = [&](const char* name) -> const std::string& {
      const std::size_t index = column.at(name);
      if (index >= fields.size()) throw std::runtime_error("short CSV row " + std::to_string(row));
      return fields[index];
    };
    try {
      Measurement m;
      m.timestamp = std::stod(get("timestamp"));
      m.sensor_id = get("sensor_id");
      m.measurement_id = std::stoi(get("measurement_id"));
      const std::string type = lowercase(get("type"));
      if (type == "cartesian" || type == "cart") m.type = MeasurementType::Cartesian;
      else if (type == "polar" || type == "radar") m.type = MeasurementType::Polar;
      else throw std::runtime_error("unknown measurement type '" + type + "'");
      m.value1 = std::stod(get("value1"));
      m.value2 = std::stod(get("value2"));
      m.true_target_id = std::stoi(get("true_target_id"));
      m.sensor_x = std::stod(get("sensor_x"));
      m.sensor_y = std::stod(get("sensor_y"));
      const double position_std = optional(fields, "position_std", 3.0);
      const double range_std = optional(fields, "range_std", 4.0);
      const double bearing_std = optional(fields, "bearing_std", 0.05);
      m.position_variance = position_std * position_std;
      m.range_variance = range_std * range_std;
      m.bearing_variance = bearing_std * bearing_std;
      if (!std::isfinite(m.timestamp) || !std::isfinite(m.value1) ||
          !std::isfinite(m.value2) || !std::isfinite(m.sensor_x) ||
          !std::isfinite(m.sensor_y) || position_std < 0.0 ||
          range_std < 0.0 || bearing_std < 0.0 ||
          (m.type == MeasurementType::Polar && m.value1 < 0.0))
        throw std::runtime_error("non-finite value, invalid standard deviation, or negative range");
      measurements.push_back(std::move(m));
    } catch (const std::exception& error) {
      throw std::runtime_error("invalid CSV row " + std::to_string(row) + ": " + error.what());
    }
  }
  std::stable_sort(measurements.begin(), measurements.end(), [](const Measurement& a, const Measurement& b) {
    return a.timestamp < b.timestamp;
  });
  return measurements;
}

void write_estimates_csv(const std::string& path, const std::vector<Estimate>& estimates) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot write estimates CSV: " + path);
  output << "timestamp,track_id,x,y,vx,vy,truth_target_id\n" << std::setprecision(12);
  for (const auto& e : estimates)
    output << e.timestamp << ',' << e.track_id << ',' << e.x << ',' << e.y << ','
           << e.vx << ',' << e.vy << ',' << e.truth_target_id << '\n';
}

}  // namespace tracker
