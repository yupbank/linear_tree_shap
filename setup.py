from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext


# Extend the default build_ext class to bootstrap numpy installation
# that are needed to build C extensions.
# see https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        if isinstance(__builtins__, dict):
            __builtins__["__NUMPY_SETUP__"] = False
        else:
            setattr(__builtins__, "__NUMPY_SETUP__", False)
        import numpy
        print("numpy.get_include()", numpy.get_include())
        self.include_dirs.append(numpy.get_include())



def run_setup():
    ext_modules = [
        Extension('linear_tree_shap._cext', sources=['linear_tree_shap/cext/_cext.cc'],
                  )]


    setup(
        name='linear_tree_shap',
        version='0.1',
        description='Tree Shap computed in linear time.',
        url='http://github.com/yupbank/linear_tree_shap',
        author='peng yu',
        author_email='yupbank@gmail.com',
        license='MIT',
        packages=[
            'linear_tree_shap'],
        package_data={'linear_tree_shap': ['cext/linear_tree_shap.h', 'cext/linear_tree_shap_v2.h']},
        cmdclass={'build_ext': build_ext},
        setup_requires=['numpy'],
        install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'tqdm>4.25.0', # numpy versions are for numba
                          'packaging>20.9', 'slicer==0.0.7', 'numba', 'cloudpickle'],
        ext_modules=ext_modules,
        classifiers=[
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
        ],
        zip_safe=False
        # python_requires='>3.0' we will add this at some point
    )


# we seem to need this import guard for appveyor
if __name__ == "__main__":
    run_setup()
