#pragma once

#include <string>
#include <vector>

#include "tracker/measurement.hpp"
#include "tracker/track_manager.hpp"

namespace tracker {

std::vector<Measurement> read_measurements_csv(const std::string& path);
void write_estimates_csv(const std::string& path, const std::vector<Estimate>& estimates);

}  // namespace tracker
