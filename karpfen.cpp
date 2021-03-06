/*
 * This file is part of Karpfen, an OpenCL poisson solver.
 *
 * Copyright (C) 2017  Aksel Alpay
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define VIENNACL_WITH_OPENCL
//#define VIENNACL_WITH_CUDA
#include "fits.hpp"
#include "karpfen.hpp"
#include <iostream>
#include <vector>
#include <viennacl/ocl/device.hpp>
#include <viennacl/ocl/backend.hpp>

#if DOUBLE_PRECISION==1
using scalar = double;
#else
using scalar = float;
#endif

void usage()
{
  std::cout << "Usage: karpfen platform=<OpenCL Platform Id> <right hand side fits file> <boundary condition fits file> <dx>\n";
}

int get_platform_index(const std::string& platform_string)
{
  std::string identifier_string = "platform=";
  if(platform_string.find(identifier_string) == std::string::npos)
    throw std::invalid_argument{"Invalid platform string"};

  std::string id_string = platform_string.substr(identifier_string.size());
  return std::stoi(id_string);
}

template<class T>
std::vector<T> resample(const std::vector<T>& data,
                        std::size_t resampled_size)
{
  std::vector<T> out(resampled_size);

  for(std::size_t i = 0; i < resampled_size; ++i)
  {
    scalar relative_position = static_cast<scalar>(i)/static_cast<scalar>(resampled_size-1);

    scalar original_position = relative_position * (data.size()-1);
    std::size_t original_position0 = static_cast<std::size_t>(original_position);
    std::size_t original_position1 = original_position0 + 1;

    assert(original_position0 < data.size());

    T v0 = data[original_position0];

    T v1 = 0.0;
    scalar f = 0.0;
    if(original_position1 < data.size())
    {
      v1 = data[original_position1];
      f = original_position - static_cast<scalar>(original_position0);
    }

    out[i] = f * v1 + (1 - f) * v0;
  }

  return out;
}

template<class T>
std::vector<T> crop_ends(const std::vector<T>& x)
{
  assert(x.size() >= 2);
  std::vector<T> result(x.size()-2);
  std::copy(x.begin()+1,x.end()-1,result.begin());
  return result;
}

int main(int argc, char** argv)
{
  if(argc != 5)
  {
    usage();
    return -1;
  }

  try
  {
#if DOUBLE_PRECISION==1
    std::cout << "Starting Karpfen [double precision]" << std::endl;
#else
    std::cout << "Starting Karpfen [single precision]" << std::endl;
#endif

    int platform = get_platform_index(argv[1]);
    viennacl::ocl::set_context_platform_index(0, platform);
    if(viennacl::ocl::platform(platform).devices().size() == 0)
      throw std::runtime_error{"No OpenCL devices available."};


    std::cout << "Using OpenCL device: " << viennacl::ocl::current_device().name() << std::endl;

    karpfen::util::fits<scalar> rhs_file{argv[2]};
    karpfen::util::fits<scalar> bc_file{argv[3]};

    karpfen::util::multi_array<scalar> rhs, bc;

    scalar dx = static_cast<scalar>(std::stod(argv[4]));
    std::cout << "Using pixel size dx=" << dx << std::endl;

    rhs_file.load(rhs);
    bc_file.load(bc);

    if(rhs.get_dimension() != 2)
      throw std::invalid_argument{"rhs file must be 2d!\n"};

    if(bc.get_dimension() != 2)
      throw std::invalid_argument{"bc file must be 2d!\n"};

    if(rhs.get_extent_of_dimension(0) <= 2 ||
       rhs.get_extent_of_dimension(1) <= 2)
      throw std::invalid_argument{"rhs must be at least 3x3 pixels in size."};

    std::size_t size_x = rhs.get_extent_of_dimension(0)-2;
    std::size_t size_y = rhs.get_extent_of_dimension(1)-2;

    std::size_t bc_size_x = bc.get_extent_of_dimension(0);
    std::size_t bc_size_y = bc.get_extent_of_dimension(1);

    // Extract boundary conditions
    std::vector<scalar> top_bc(bc_size_x);
    std::vector<scalar> bottom_bc(bc_size_x);
    std::vector<scalar> left_bc(bc_size_y);
    std::vector<scalar> right_bc(bc_size_y);

    for(std::size_t x = 0; x < bc_size_x; ++x)
    {
      std::size_t bottom_idx[] = {x, 0};
      std::size_t top_idx[]    = {x, bc_size_y-1};
      top_bc[x]    = bc[top_idx];
      bottom_bc[x] = bc[bottom_idx];
    }
    for(std::size_t y = 0; y < bc_size_y; ++y)
    {
      std::size_t left_idx[] = {0,           y};
      std::size_t right_idx[]= {bc_size_x-1, y};
      left_bc[y]  = bc[left_idx];
      right_bc[y] = bc[right_idx];
    }
    // ToDo: Only crop ends if rhs_size == bc_size
    top_bc    = resample(crop_ends(top_bc),    size_x);
    bottom_bc = resample(crop_ends(bottom_bc), size_x);
    left_bc   = resample(crop_ends(left_bc),   size_y);
    right_bc  = resample(crop_ends(right_bc),  size_y);
    // Crop boundary layer from rhs
    karpfen::util::multi_array<scalar> cropped_rhs{size_x, size_y};
    for(std::size_t x = 0; x < size_x; ++x)
      for(std::size_t y = 0;  y < size_y; ++y)
      {
        std::size_t cropped_idx[]   = {x  ,y  };
        std::size_t uncropped_idx[] = {x+1,y+1};
        cropped_rhs[cropped_idx] = rhs[uncropped_idx];
      }

    karpfen::util::multi_array<scalar> solution;
    // Solve system
    karpfen::dirichlet_system2d<scalar> system{&cropped_rhs,
          left_bc,
          right_bc,
          top_bc,
          bottom_bc,
          dx};
    karpfen::cg_solver<scalar> solver{1.0e-6f, 3000};

    std::cout << "Solving sytem..." << std::endl;
    system.solve(solver, solution);
    std::cout << "Solved system in " << solver.get_num_iterations()
              << " iterations (error = " << solver.get_error() << ")"
              << std::endl;


    // Combine boundary layer and solution to output
    karpfen::util::multi_array<scalar> result(rhs.get_extent_of_dimension(0),
                                              rhs.get_extent_of_dimension(1));

    for(std::size_t x = 0; x < size_x; ++x)
    {
      std::size_t top_boundary []    = {x+1, size_y+1};
      std::size_t bottom_boundary [] = {x+1, 0};
      result[top_boundary] = top_bc[x];
      result[bottom_boundary] = bottom_bc[x];

      for(std::size_t y = 0;  y < size_y; ++y)
      {
        std::size_t cropped_idx[]   = {x  ,y  };
        std::size_t uncropped_idx[] = {x+1,y+1};
        result[uncropped_idx] = solution[cropped_idx];
      }
    }
    for(std::size_t y = 0; y < size_y; ++y)
    {
      std::size_t left_boundary [] = {0       , y+1};
      std::size_t right_boundary[] = {size_x+1, y+1};
      result[left_boundary] = left_bc[y];
      result[right_boundary] = right_bc[y];
    }

    karpfen::util::fits<scalar> result_file{"karpfen_output.fits"};
    result_file.save(result);

  }
  catch(std::exception& e)
  {
    std::cout << "Error: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}

