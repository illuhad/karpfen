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


#ifndef KARPFEN_HPP
#define KARPFEN_HPP

#include <memory>
#include <cassert>
#include <map>
#include <algorithm>
#include <boost/container/flat_map.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/amg.hpp>
#include "multi_array.hpp"

namespace karpfen {

template<typename Scalar>
class system
{
public:
  using sparse_matrix_type = viennacl::compressed_matrix<Scalar>;
  using dense_vector_type = viennacl::vector<Scalar>;

  virtual std::unique_ptr<sparse_matrix_type> assemble_system_matrix() const = 0;
  virtual std::unique_ptr<dense_vector_type>  assemble_rhs() const = 0;
};

template<typename Scalar>
class system2d : public system<Scalar>
{
public:
  using typename system<Scalar>::dense_vector_type;
  using typename system<Scalar>::sparse_matrix_type;

  system2d(const util::multi_array<Scalar>* f,
           Scalar dx)
    : _f{f}, _dx{dx}
  {
    assert(f != nullptr);
    assert(f->get_dimension() == 2);
    assert(_dx > 0.0f);

    this->_num_dofs_x = f->get_extent_of_dimension(0);
    this->_num_dofs_y = f->get_extent_of_dimension(1);
    this->_num_dofs = _num_dofs_x * _num_dofs_y;

    assert(_num_dofs > 0);
  }

  virtual ~system2d(){}

  virtual std::unique_ptr<sparse_matrix_type> assemble_system_matrix() const override
  {
    std::vector<std::map<unsigned, Scalar>> host_system_matrix(this->_num_dofs);

    for(std::size_t y = 0; y < this->_num_dofs_y; ++y)
    {
      for(std::size_t x = 0; x < this->_num_dofs_x; ++x)
      {
        std::size_t dof_id = get_dof_id(x,y);

        if(x > 0)
        {
          std::size_t stencil_xm1 = get_dof_id(x-1,y);
          host_system_matrix[dof_id][stencil_xm1] = -1.0f;
        }
        if(x < _num_dofs_x - 1)
        {
          std::size_t stencil_x1  = get_dof_id(x+1,y);
          host_system_matrix[dof_id][stencil_x1] = -1.0f;
        }

        if(y > 0)
        {
          std::size_t stencil_ym1 = get_dof_id(x,y-1);
          host_system_matrix[dof_id][stencil_ym1] = -1.0f;
        }
        if(y < _num_dofs_y - 1)
        {
          std::size_t stencil_y1  = get_dof_id(x,y+1);
          host_system_matrix[dof_id][stencil_y1] = -1.0;
        }

        host_system_matrix[dof_id][dof_id] = 4.0f;
      }
    }

    std::unique_ptr<sparse_matrix_type> matrix{new sparse_matrix_type(_num_dofs, _num_dofs)};
    viennacl::copy(host_system_matrix, *matrix);

    return matrix;
  }

  virtual std::unique_ptr<dense_vector_type> assemble_rhs() const override
  {
    assert(this->get_num_dofs() == _f->get_num_elements());

    std::unique_ptr<dense_vector_type> rhs{new dense_vector_type(this->get_num_dofs())};

    viennacl::copy(_f->begin(), _f->end(), rhs->begin());
    // Multiply by dx^2 factor
    (*rhs) *= -(_dx*_dx);
    return rhs;
  }

  std::size_t get_num_dofs() const
  {
    return _num_dofs;
  }

  std::size_t get_num_dofs_x() const
  {
    return _num_dofs_x;
  }

  std::size_t get_num_dofs_y() const
  {
    return _num_dofs_y;
  }

  Scalar get_dx() const
  {
    return _dx*_dx;
  }

  template<class Solver>
  void solve(const Solver& s, util::multi_array<Scalar>& result) const
  {
    dense_vector_type gpu_result = s.solve(*this);

    result = util::multi_array<Scalar>{_num_dofs_x, _num_dofs_y};
    viennacl::copy(gpu_result.begin(), gpu_result.end(), result.begin());
  }

protected:
  const util::multi_array<Scalar>* _f;

  std::size_t get_dof_id(std::size_t x, std::size_t y) const
  {
    assert(x < _num_dofs_x);
    assert(y < _num_dofs_y);
    return y * this->_num_dofs_x + x;
  }

private:
  std::size_t _num_dofs_x;
  std::size_t _num_dofs_y;
  Scalar _dx;
  std::size_t _num_dofs;
};

template<typename Scalar>
class dirichlet_system2d : public system2d<Scalar>
{
public:
  using typename system<Scalar>::dense_vector_type;
  using typename system<Scalar>::sparse_matrix_type;

  dirichlet_system2d(const util::multi_array<Scalar>* f,
                     const std::vector<Scalar>& left_bc,
                     const std::vector<Scalar>& right_bc,
                     const std::vector<Scalar>& top_bc,
                     const std::vector<Scalar>& bottom_bc,
                     Scalar dx)
    : system2d<Scalar>{f, dx},
      _left_bc{left_bc},
      _right_bc{right_bc},
      _top_bc{top_bc},
      _bottom_bc{bottom_bc}
  {
    assert(left_bc.size() == this->get_num_dofs_y());
    assert(right_bc.size() == this->get_num_dofs_y());

    assert(top_bc.size() == this->get_num_dofs_x());
    assert(bottom_bc.size() == this->get_num_dofs_x());
  }

  virtual ~dirichlet_system2d(){}

  virtual std::unique_ptr<dense_vector_type> assemble_rhs() const override
  {
    assert(this->_f->get_num_elements() == this->get_num_dofs());

    std::unique_ptr<dense_vector_type> rhs{new dense_vector_type(this->get_num_dofs())};

    std::vector<Scalar> host_rhs(this->get_num_dofs());
    std::copy(this->_f->begin(), this->_f->end(), host_rhs.begin());

    // Apply dx^2 factor
    Scalar dx2 = this->get_dx()*this->get_dx();
    for(std::size_t i = 0; i < host_rhs.size(); ++i)
      host_rhs[i] *= -dx2;

    // Add dirichlet BCs

    // Iterate over left and right edge
    for(std::size_t y = 0; y < this->get_num_dofs_y(); ++y)
    {
      host_rhs[this->get_dof_id(0                       , y)] += _left_bc[y];
      host_rhs[this->get_dof_id(this->get_num_dofs_x()-1, y)] += _right_bc[y];
    }

    // Iterate over top and bottom edge
    for(std::size_t x = 0; x < this->get_num_dofs_x(); ++x)
    {
      host_rhs[this->get_dof_id(x, 0                       )] += _top_bc[x];
      host_rhs[this->get_dof_id(x, this->get_num_dofs_y()-1)] += _bottom_bc[x];
    }

    viennacl::copy(host_rhs, *rhs);
    return rhs;
  }

private:
  std::vector<Scalar> _left_bc;
  std::vector<Scalar> _right_bc;
  std::vector<Scalar> _top_bc;
  std::vector<Scalar> _bottom_bc;
};

template<class Scalar>
class cg_solver
{
public:

  using dense_vector_type = typename system<Scalar>::dense_vector_type;

  cg_solver(Scalar tolerance=1.0e-8f, unsigned max_iterations=300)
    : _tol{tolerance},
      _max_iterations{max_iterations}
  {}

  dense_vector_type solve(const system<Scalar>& sys) const
  {
    auto matrix_ptr = sys.assemble_system_matrix();
    auto rhs_ptr    = sys.assemble_rhs();


    viennacl::linalg::amg_tag precond_tag;
    viennacl::linalg::amg_precond<viennacl::compressed_matrix<Scalar>> amg(*matrix_ptr, precond_tag);
    amg.setup();

    viennacl::linalg::cg_tag tag{_tol, _max_iterations};
    viennacl::linalg::cg_solver<dense_vector_type> solver{tag};

    auto result = solver(*matrix_ptr, *rhs_ptr, amg);
    _iterations = solver.tag().iters();
    _error = solver.tag().error();

    return result;
  }

  unsigned get_num_iterations() const
  {
    return _iterations;
  }

  Scalar get_error() const
  {
    return _error;
  }
private:
  Scalar _tol;
  unsigned _max_iterations;

  mutable unsigned _iterations = 0;
  mutable Scalar _error = 0.0f;
};

}

#endif
