#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geoserver_msgs/srv/get_map_tif.hpp>
#include <geoserver_msgs/srv/get_map_height_at.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <cpl_conv.h>
#include <curl/curl.h>
#include <optional>
#include <limits>

namespace fs = std::filesystem;

class GeoServerMapTifNode : public rclcpp::Node
{
public:
  GeoServerMapTifNode()
  : Node("geoserver_map_tif_node")
  {
    // Declare parameters
    this->declare_parameter("geoserver_url", "http://localhost:8080/geoserver/wcs");
    this->declare_parameter("coverage_id", "ne__ALS_DTM_CRS3035RES50000mN2650000E4700000");
    this->declare_parameter("preview_topic", "/map/tif_preview");


    geoserver_url_ = this->get_parameter("geoserver_url").as_string();
    coverage_id_ = this->get_parameter("coverage_id").as_string();
    preview_topic_ = this->get_parameter("preview_topic").as_string();

    // Preview publisher
    preview_pub_ = this->create_publisher<sensor_msgs::msg::Image>(preview_topic_, 10);

    // Services
    map_tif_srv_ = this->create_service<geoserver_msgs::srv::GetMapTif>(
      "/map/get_map_tif",
      std::bind(&GeoServerMapTifNode::handle_get_map_tif, this,
                std::placeholders::_1, std::placeholders::_2));

    height_srv_ = this->create_service<geoserver_msgs::srv::GetMapHeightAt>(
      "/map/get_height_at",
      std::bind(&GeoServerMapTifNode::handle_get_height_at, this,
                std::placeholders::_1, std::placeholders::_2));

    // Initialize GDAL
    GDALAllRegister();

    // Initialize OGR for coordinate transformation
    OGRRegisterAll();

    // Setup coordinate transformation: WGS84 (Lat/Lon) -> EPSG:3035
    source_srs_ = std::make_unique<OGRSpatialReference>();
    target_srs_ = std::make_unique<OGRSpatialReference>();
    
    // WGS84: latitude/longitude in degrees
    if (source_srs_->importFromEPSG(4326) != OGRERR_NONE) {
      RCLCPP_ERROR(this->get_logger(), "Failed to import WGS84 (EPSG:4326)!");
    }
    
    // EPSG:3035: ETRS89 / LAEA Europe (meters)
    // Note: EPSG:3035 uses (Easting, Northing) = (X, Y) order
    if (target_srs_->importFromEPSG(3035) != OGRERR_NONE) {
      RCLCPP_ERROR(this->get_logger(), "Failed to import EPSG:3035!");
    }
    
    // Set axis mapping strategy to traditional (Easting=X, Northing=Y)
    source_srs_->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    target_srs_->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    
    coord_transform_ = OGRCreateCoordinateTransformation(source_srs_.get(), target_srs_.get());
    
    if (coord_transform_ == nullptr) {
      RCLCPP_ERROR(this->get_logger(), "Failed to create coordinate transformation!");
    } else {
      RCLCPP_INFO(this->get_logger(), "Coordinate transformation WGS84->EPSG:3035 initialized.");
      
      // Test transformation with known Graz coordinates
      double test_lon = 15.4393;
      double test_lat = 47.0745;
      double test_x = test_lon;
      double test_y = test_lat;
      double test_z = 0.0;
      if (coord_transform_->Transform(1, &test_x, &test_y, &test_z)) {
        RCLCPP_INFO(this->get_logger(), "Test transform: WGS84(%.6f, %.6f) -> EPSG:3035(%.2f, %.2f)", 
                    test_lon, test_lat, test_x, test_y);
      }
    }

    // Initialize curl
    curl_global_init(CURL_GLOBAL_DEFAULT);

    RCLCPP_INFO(this->get_logger(), "GeoServerMapTifNode started.");
  }

  ~GeoServerMapTifNode()
  {
    if (current_dataset_ != nullptr) {
      GDALClose(current_dataset_);
      current_dataset_ = nullptr;
    }
    if (coord_transform_ != nullptr) {
      OCTDestroyCoordinateTransformation(coord_transform_);
      coord_transform_ = nullptr;
    }
    curl_global_cleanup();
  }

private:
  // Callback for writing HTTP response data
  static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
  {
    ((std::string *)userp)->append((char *)contents, size * nmemb);
    return size * nmemb;
  }

  void handle_get_map_tif(
    const std::shared_ptr<geoserver_msgs::srv::GetMapTif::Request> request,
    std::shared_ptr<geoserver_msgs::srv::GetMapTif::Response> response)
  {
    // Input: x = latitude (degrees), y = longitude (degrees) in WGS84
    // Note: Standard convention is (lat, lon) but we use (x, y) for consistency
    double lat = request->x;  // latitude in degrees
    double lon = request->y;  // longitude in degrees
    double size = request->size_m;
    double res = request->resolution_m;
    std::string out_path = request->output_path;

    if (size <= 0.0 || res <= 0.0) {
      response->success = false;
      response->message = "size_m and resolution_m must be > 0.";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    // Transform center point from WGS84 (Lat/Lon) to EPSG:3035
    // OGR Transform expects (longitude, latitude) order for input
    double x_3035 = lon;  // longitude first (X axis)
    double y_3035 = lat;  // latitude second (Y axis)
    double z = 0.0;
    
    if (coord_transform_ == nullptr) {
      response->success = false;
      response->message = "Coordinate transformation not initialized.";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      return;
    }
    
    // Transform: input is (lon, lat) in WGS84, output is (x, y) in EPSG:3035
    // OGR Transform returns (Easting, Northing) for projected coordinates
    int transform_result = coord_transform_->Transform(1, &x_3035, &y_3035, &z);
    
    if (!transform_result) {
      response->success = false;
      response->message = "Failed to transform coordinates from WGS84 to EPSG:3035.";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      RCLCPP_ERROR(this->get_logger(), "Input: lon=%.10f, lat=%.10f", lon, lat);
      return;
    }
    

    double easting = x_3035;   // Easting (X coordinate in EPSG:3035)
    double northing = y_3035;  // Northing (Y coordinate in EPSG:3035)
    
    RCLCPP_INFO(this->get_logger(), "OGR returned: x=%.2f, y=%.2f -> Swapped to Easting=%.2f, Northing=%.2f", 
                x_3035, y_3035, easting, northing);
    RCLCPP_INFO(this->get_logger(), "Transformed: WGS84(lon=%.10f, lat=%.10f) -> EPSG:3035(Easting=%.2f, Northing=%.2f)", 
                lon, lat, easting, northing);

    // Calculate bounding box in EPSG:3035 (meters)
    double half = size / 2.0;
    double minx = easting - half;
    double maxx = easting + half;
    double miny = northing - half;
    double maxy = northing + half;

    // Build WCS request URL - exactly like the working curl command
    // Format: http://localhost:8080/geoserver/wcs?service=WCS&version=2.0.1&request=GetCoverage&coverageId=...&subset=X(...)&subset=Y(...)&format=image/tiff
    std::ostringstream url_stream;
    url_stream << std::fixed << std::noshowpoint << std::setprecision(0);
    url_stream << geoserver_url_
               << "?service=WCS&version=2.0.1&request=GetCoverage"
               << "&coverageId=" << coverage_id_
               << "&subset=X(" << static_cast<long long>(minx) << "," << static_cast<long long>(maxx) << ")"
               << "&subset=Y(" << static_cast<long long>(miny) << "," << static_cast<long long>(maxy) << ")"
               << "&format=image/tiff";
    std::string url = url_stream.str();

    RCLCPP_INFO(this->get_logger(), "WCS Request: %s", url.c_str());

    // Execute HTTP request using libcurl
    CURL *curl = curl_easy_init();
    std::string readBuffer;
    long http_code = 0;

    if (curl) {
      // Use the URL exactly as built (like curl command line does)
      // The URL contains parentheses which are valid in query strings
      // curl will handle the URL as-is, just like the command line curl does
      curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
      curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
      curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
      curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
      curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

      CURLcode res = curl_easy_perform(curl);
      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
      
      if (res != CURLE_OK) {
        std::string msg = "CURL error during WCS request: " + std::string(curl_easy_strerror(res));
        RCLCPP_ERROR(this->get_logger(), "%s", msg.c_str());
        curl_easy_cleanup(curl);
        response->success = false;
        response->message = msg;
        return;
      }
      
      curl_easy_cleanup(curl);

      if (http_code != 200) {
        std::string msg = "HTTP error during WCS request: HTTP " + std::to_string(http_code);
        RCLCPP_ERROR(this->get_logger(), "%s", msg.c_str());
        RCLCPP_ERROR(this->get_logger(), "Response size: %zu bytes", readBuffer.size());
        if (!readBuffer.empty() && readBuffer.size() < 10000) {
          RCLCPP_ERROR(this->get_logger(), "Response content: %s", readBuffer.c_str());
        }
        response->success = false;
        response->message = msg;
        return;
      }
    } else {
      std::string msg = "Failed to initialize curl";
      RCLCPP_ERROR(this->get_logger(), "%s", msg.c_str());
      response->success = false;
      response->message = msg;
      return;
    }

    // Ensure output directory exists
    try {
      fs::path out_file_path(out_path);
      fs::create_directories(out_file_path.parent_path());
    } catch (const std::exception & e) {
      std::string msg = "Failed to create output directory: " + std::string(e.what());
      RCLCPP_ERROR(this->get_logger(), "%s", msg.c_str());
      response->success = false;
      response->message = msg;
      return;
    }

    // Write GeoTIFF
    try {
      std::ofstream out_file(out_path, std::ios::binary);
      if (!out_file) {
        throw std::runtime_error("Failed to open output file");
      }
      out_file.write(readBuffer.c_str(), readBuffer.size());
      out_file.close();
    } catch (const std::exception & e) {
      std::string msg = "Failed to write GeoTIFF: " + std::string(e.what());
      RCLCPP_ERROR(this->get_logger(), "%s", msg.c_str());
      response->success = false;
      response->message = msg;
      return;
    }

    RCLCPP_INFO(this->get_logger(), "GeoTIFF saved to: %s", out_path.c_str());

    // Load TIF into memory
    if (!load_current_tif(out_path)) {
      response->success = false;
      response->message = "GeoTIFF saved but failed to load into internal state.";
      return;
    }

    // Publish preview
    publish_preview();

    response->success = true;
    response->message = "TIF saved and loaded: " + out_path;
  }

  bool load_current_tif(const std::string & path)
  {
    // Close previous dataset if open
    if (current_dataset_ != nullptr) {
      GDALClose(current_dataset_);
      current_dataset_ = nullptr;
    }

    // Open dataset
    current_dataset_ = static_cast<GDALDataset *>(GDALOpen(path.c_str(), GA_ReadOnly));
    if (current_dataset_ == nullptr) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open TIF: %s", CPLGetLastErrorMsg());
      return false;
    }

    // Get raster band
    GDALRasterBand * band = current_dataset_->GetRasterBand(1);
    if (band == nullptr) {
      RCLCPP_ERROR(this->get_logger(), "Failed to get raster band");
      GDALClose(current_dataset_);
      current_dataset_ = nullptr;
      return false;
    }

    // Get dimensions
    int width = current_dataset_->GetRasterXSize();
    int height = current_dataset_->GetRasterYSize();

    // Read data
    current_array_ = std::vector<double>(width * height);
    CPLErr err = band->RasterIO(GF_Read, 0, 0, width, height,
                                current_array_.data(), width, height,
                                GDT_Float64, 0, 0);
    if (err != CE_None) {
      RCLCPP_ERROR(this->get_logger(), "Failed to read TIF band: %s", CPLGetLastErrorMsg());
      GDALClose(current_dataset_);
      current_dataset_ = nullptr;
      return false;
    }

    // Get transform and nodata
    current_dataset_->GetGeoTransform(current_transform_);
    current_nodata_ = band->GetNoDataValue();
    current_width_ = width;
    current_height_ = height;

    current_tif_path_ = path;

    RCLCPP_INFO(this->get_logger(), "TIF loaded: %s, size=%dx%d, nodata=%f",
                path.c_str(), width, height,
                current_nodata_ ? *current_nodata_ : std::numeric_limits<double>::quiet_NaN());

    return true;
  }

  void publish_preview()
  {
    if (current_array_.empty() || current_dataset_ == nullptr) {
      RCLCPP_WARN(this->get_logger(), "No TIF loaded, cannot publish preview.");
      return;
    }

    // Find valid range
    double vmin = std::numeric_limits<double>::max();
    double vmax = std::numeric_limits<double>::lowest();
    bool has_valid = false;

    for (size_t i = 0; i < current_array_.size(); ++i) {
      double val = current_array_[i];
      if (std::isfinite(val) && 
          (!current_nodata_ || (!std::isnan(*current_nodata_) && 
                                !std::isnan(val) && 
                                std::abs(val - *current_nodata_) > 1e-9))) {
        vmin = std::min(vmin, val);
        vmax = std::max(vmax, val);
        has_valid = true;
      }
    }

    if (!has_valid) {
      RCLCPP_WARN(this->get_logger(), "Preview failed: all values are invalid.");
      return;
    }

    if (std::abs(vmin - vmax) < 1e-9) {
      RCLCPP_WARN(this->get_logger(), "Preview scaling impossible: vmin == vmax.");
      return;
    }

    // Normalize to 0..255
    cv::Mat img_float(current_height_, current_width_, CV_64F);
    cv::Mat img_u8(current_height_, current_width_, CV_8UC1);

    for (int y = 0; y < current_height_; ++y) {
      for (int x = 0; x < current_width_; ++x) {
        double val = current_array_[y * current_width_ + x];
        bool is_valid = std::isfinite(val) && 
                       (!current_nodata_ || (!std::isnan(*current_nodata_) && 
                                             !std::isnan(val) && 
                                             std::abs(val - *current_nodata_) > 1e-9));

        if (is_valid) {
          double scaled = (val - vmin) / (vmax - vmin);
          scaled = std::max(0.0, std::min(1.0, scaled));
          img_u8.at<uchar>(y, x) = static_cast<uchar>(scaled * 255.0);
        } else {
          img_u8.at<uchar>(y, x) = 0;
        }
      }
    }

    // Apply viridis colormap
    cv::Mat colored;
    cv::applyColorMap(img_u8, colored, cv::COLORMAP_VIRIDIS);

    // Limit preview size to 2000x2000px while preserving aspect ratio
    const int max_preview_size = 2000;
    cv::Mat preview_img = colored;
    if (colored.cols > max_preview_size || colored.rows > max_preview_size) {
      double scale = std::min(
        static_cast<double>(max_preview_size) / colored.cols,
        static_cast<double>(max_preview_size) / colored.rows
      );
      int new_width = static_cast<int>(colored.cols * scale);
      int new_height = static_cast<int>(colored.rows * scale);
      cv::resize(colored, preview_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);
      RCLCPP_INFO(this->get_logger(), "Preview resized from %dx%d to %dx%d",
                  colored.cols, colored.rows, new_width, new_height);
    }

    // Convert to ROS message
    sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(
      std_msgs::msg::Header(), "bgr8", preview_img).toImageMsg();
    msg->header.stamp = this->now();
    msg->header.frame_id = "map";

    preview_pub_->publish(*msg);
    RCLCPP_INFO(this->get_logger(), "Color preview (viridis) published on %s (size=%dx%d)",
                preview_topic_.c_str(), preview_img.cols, preview_img.rows);
  }

  void handle_get_height_at(
    const std::shared_ptr<geoserver_msgs::srv::GetMapHeightAt::Request> request,
    std::shared_ptr<geoserver_msgs::srv::GetMapHeightAt::Response> response)
  {
    if (current_dataset_ == nullptr || current_array_.empty()) {
      std::string msg = "No GeoTIFF loaded. Call /map/get_map_tif first.";
      RCLCPP_WARN(this->get_logger(), "%s", msg.c_str());
      response->success = false;
      response->height = std::numeric_limits<double>::quiet_NaN();
      response->message = msg;
      return;
    }

    // Input: x = latitude (degrees), y = longitude (degrees) in WGS84
    // Note: Standard convention is (lat, lon) but we use (x, y) for consistency
    double lat = request->x;  // latitude in degrees
    double lon = request->y;  // longitude in degrees
    
    // Transform center point from WGS84 (Lat/Lon) to EPSG:3035
    // OGR Transform expects (longitude, latitude) order for input
    double x_3035 = lon;  // longitude first (X axis)
    double y_3035 = lat;  // latitude second (Y axis)
    double z = 0.0;
    
    if (coord_transform_ == nullptr) {
      response->success = false;
      response->height = std::numeric_limits<double>::quiet_NaN();
      response->message = "Coordinate transformation not initialized.";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      return;
    }
    
    // Transform: input is (lon, lat) in WGS84, output is (x, y) in EPSG:3035
    // OGR Transform returns (Easting, Northing) for projected coordinates
    int transform_result = coord_transform_->Transform(1, &x_3035, &y_3035, &z);
    
    if (!transform_result) {
      response->success = false;
      response->height = std::numeric_limits<double>::quiet_NaN();
      response->message = "Failed to transform coordinates from WGS84 to EPSG:3035.";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      RCLCPP_ERROR(this->get_logger(), "Input: lon=%.10f, lat=%.10f", lon, lat);
      return;
    }
    
    // OGR Transform returns coordinates for EPSG:3035
    double x = x_3035;   // Easting (X coordinate in EPSG:3035)
    double y = y_3035;  // Northing (Y coordinate in EPSG:3035)

    // Transform geographic coordinates to pixel coordinates
    // GDAL transform: [origin_x, pixel_width, rotation, origin_y, rotation, pixel_height]
    // For north-up images: pixel_height is negative
    // pixel_x = (geo_x - origin_x) / pixel_width
    // pixel_y = (geo_y - origin_y) / pixel_height
    double pixel_x = (x - current_transform_[0]) / current_transform_[1];
    double pixel_y = (y - current_transform_[3]) / current_transform_[5];

    // Check bounds
    int px = static_cast<int>(std::round(pixel_x));
    int py = static_cast<int>(std::round(pixel_y));

    if (px < 0 || px >= current_width_ || py < 0 || py >= current_height_) {
      std::string msg = "Point is outside the loaded GeoTIFF bounds";
      RCLCPP_WARN(this->get_logger(), "%s", msg.c_str());
      response->success = false;
      response->height = std::numeric_limits<double>::quiet_NaN();
      response->message = msg;
      return;
    }

    // Get height value
    double height = current_array_[py * current_width_ + px];

    // Handle NoData
    if (current_nodata_ && std::isfinite(*current_nodata_) && 
        std::isfinite(height) && std::abs(height - *current_nodata_) < 1e-9) {
      std::string msg = "Point lies in NoData region.";
      RCLCPP_WARN(this->get_logger(), "%s", msg.c_str());
      response->success = false;
      response->height = std::numeric_limits<double>::quiet_NaN();
      response->message = msg;
      return;
    }

    // Normal success
    response->success = true;
    response->height = height;
    response->message = "OK";
  }

  // Parameters
  std::string geoserver_url_;
  std::string coverage_id_;
  std::string preview_topic_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr preview_pub_;

  // Services
  rclcpp::Service<geoserver_msgs::srv::GetMapTif>::SharedPtr map_tif_srv_;
  rclcpp::Service<geoserver_msgs::srv::GetMapHeightAt>::SharedPtr height_srv_;

  // Internal state for the loaded DEM
  std::string current_tif_path_;
  GDALDataset * current_dataset_ = nullptr;
  std::vector<double> current_array_;
  double current_transform_[6];
  std::optional<double> current_nodata_;
  int current_width_ = 0;
  int current_height_ = 0;
  
  // Coordinate transformation
  std::unique_ptr<OGRSpatialReference> source_srs_;
  std::unique_ptr<OGRSpatialReference> target_srs_;
  OGRCoordinateTransformation * coord_transform_ = nullptr;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GeoServerMapTifNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

