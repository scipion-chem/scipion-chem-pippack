# **************************************************************************
# *
# * Authors:     Blanca Pueche (blanca.pueche@cnb.csic.es)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import os
from os.path import join, exists

import pwem
from scipion.install.funcs import InstallHelper

from pyworkflow import SPA, TOMO, MODELLING
from .constants import PIPPACK_DIC
from pwchem import Plugin as pwchemPlugin

_version_ = '0.1'
_references = ['']


class Plugin(pwem.Plugin):
    _homeVar = PIPPACK_DIC['home']
    _pathVars = [PIPPACK_DIC['home']]
    _supportedVersions = [PIPPACK_DIC['version']]

    @classmethod
    def _defineVariables(cls):
        """ Return and write a variable in the config file.
        """
        cls._defineEmVar(PIPPACK_DIC['home'], PIPPACK_DIC['name'] + '-' + PIPPACK_DIC['version'])

    @classmethod
    def defineBinaries(cls, env, default=True):
        installer = InstallHelper(
            PIPPACK_DIC['name'],
            packageHome=cls.getVar(PIPPACK_DIC['home']),
            packageVersion=PIPPACK_DIC['version']
        )

        installer.getCondaEnvCommand(
            PIPPACK_DIC['name'],
            binaryVersion=PIPPACK_DIC['version'],
            pythonVersion='3.11'
        ).addCommand(
            f"{pwchemPlugin.getEnvActivationCommand(PIPPACK_DIC)} && "
            "git clone https://github.com/Kuhlman-Lab/PIPPack.git "
            f"{join(cls.getVar(PIPPACK_DIC['home']), 'PIPPack')}",
            "REPOSITORY_CLONED"

        ).addCommand(
            f"{pwchemPlugin.getEnvActivationCommand(PIPPACK_DIC)} && "
            "conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 "
            "-c pytorch -c nvidia",
            "PYTORCH_INSTALLED"
        ).addCommand(
            f"{pwchemPlugin.getEnvActivationCommand(PIPPACK_DIC)} && "
            "conda install -y mkl=2024.0.0",
            "MKL_INSTALLED"
        ).addCommand(
            f"{pwchemPlugin.getEnvActivationCommand(PIPPACK_DIC)} && "
            "conda install -y lightning=2.0.1 -c conda-forge",
            "LIGHTNING_INSTALLED"
        ).addCommand(
            f"{pwchemPlugin.getEnvActivationCommand(PIPPACK_DIC)} && "
            "python -m pip install 'setuptools<81'",
            "SETUPTOOLS_INSTALLED"
        ).addCommand(
            f"{pwchemPlugin.getEnvActivationCommand(PIPPACK_DIC)} && "
            "python -m pip install -U torch-geometric biopython hydra-core",
            "PIP_DEPENDENCIES_INSTALLED"
        ).addPackage(
            env,
            dependencies=['git', 'conda'],
            default=default
        )

