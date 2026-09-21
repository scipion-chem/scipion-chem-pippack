# **************************************************************************
# *
# * Authors:   Blanca Pueche (blanca.pueche@cnb.csis.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
import json
import shutil

import os, json
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
import pyworkflow.protocol.params as params
from pyworkflow.protocol.constants import LEVEL_ADVANCED
from pwem.protocols import EMProtocol
from pyworkflow.object import String, Float

from pippack.__init__ import Plugin
from pwchem.__init__ import Plugin as pwchemPlugin
from pwem.objects.data import AtomStruct, SetOfAtomStructs
from pwchem.objects.base import SmallMolecule, SetOfSmallMolecules
from pwem.convert import cifToPdb

from pippack.constants import PIPPACK_DIC
from pwchem.constants import OPENBABEL_DIC


class ProtPIPPack(EMProtocol):
    """
 Generates protein side-chain conformations from an input protein
        structure or ensemble using PIPPack.

        The protocol performs side-chain prediction with PIPPack, allowing
        sampling of alternative chi-angle conformations and optional
        post-prediction resampling to reduce steric clashes. The protocol can
        process either a single protein structure or a SetOfAtomStructs
        containing multiple structures.

        Workflow
        --------
        1. Receive either a single AtomStruct or a SetOfAtomStructs as input.
        2. Convert input structures to PDB format when necessary.
        3. Prepare the input PDB files in the protocol working directory.
        4. Locate the PIPPack installation and its pretrained model weights.
        5. Run the PIPPack inference workflow using the selected parameters.
        6. Optionally use GPU acceleration for PIPPack execution.
        7. Optionally perform post-prediction resampling to reduce steric
           clashes between side chains.
        8. Collect the generated PDB structures.
        9. Return a single AtomStruct when one structure is generated, or a
           SetOfAtomStructs when multiple structures are produced.

        Input
        -----
        - inputType:
            Selects whether the protocol processes a single structure or an
            ensemble of structures.

            Available options are:
            - AtomStruct
            - SetOfAtomStructs

        - inputStruct:
            Input AtomStruct containing the protein structure to be processed.
            Used when ``inputType`` is set to AtomStruct.

        - inputEnsemble:
            SetOfAtomStructs containing the protein structures to be processed.
            Used when ``inputType`` is set to SetOfAtomStructs.

        Parameters
        ----------
        - Chi temperature:
            Temperature used by PIPPack when sampling side-chain chi angles.

            The default value is 1.0. Higher or lower values can modify the
            sampling behaviour of alternative side-chain conformations.

        - Recycles:
            Number of recycling iterations performed during PIPPack inference.

            The default value is 3.

        - Use resampling:
            Enables the post-prediction resampling procedure used to reduce
            steric clashes between residues.

            The default value is enabled.

        - Sampling temperature:
            Temperature used when sampling alternative side-chain conformations
            during the resampling procedure.

            The default value is 0.1.

        - Clash overlap tolerance:
            Tolerance for atomic overlap when evaluating steric clashes during
            resampling.

            The default value is 0.4.

        - Proline tolerance factor:
            Tolerance factor applied to proline residues during clash
            evaluation.

            The default value is 12.

        - Maximum iterations:
            Maximum number of iterations performed by the resampling procedure.

            The default value is 50.

        - Metropolis temperature:
            Temperature controlling the acceptance of candidate conformations
            during the Metropolis-based resampling procedure.

            The default value is 0.000005.

        - Use GPU for execution:
            Enables GPU acceleration for PIPPack.

            The default value is enabled.

        - Choose GPU IDs:
            Comma-separated list of GPU devices available for execution.

            The default GPU ID is 0.

        Side-Chain Prediction
        ---------------------
        PIPPack is used to predict protein side-chain conformations from the
        input structures.

        The protocol provides control over the chi-angle sampling temperature
        and the number of recycling iterations used during inference.

        Resampling
        ----------
        When enabled, PIPPack performs an additional resampling procedure after
        the initial prediction.

        The resampling procedure evaluates steric clashes and samples
        alternative side-chain conformations using the following parameters:

        - Sampling temperature
        - Clash overlap tolerance
        - Proline tolerance factor
        - Maximum iterations
        - Metropolis temperature

        This step is intended to reduce the number of clashing residues in the
        predicted structures.

        Output
        ------
        - outputAtomStruct:
            Generated AtomStruct when the PIPPack workflow produces a single
            PDB structure.

        - outputAtomStructs:
            SetOfAtomStructs containing the generated structures when multiple
            PDB files are produced.

        The output structures are stored in PDB format. If the input consists
        of multiple structures, each generated PDB is retained as an individual
        AtomStruct in the output set.

        Summary
        -------
        The protocol generates protein structures with predicted side-chain
        conformations using PIPPack.

        Depending on the number of generated structures, the output is provided
        either as a single AtomStruct or as a SetOfAtomStructs.

        Use Cases
        ---------
        - Predicting protein side-chain conformations
        - Completing or refining protein structural models
        - Generating side-chain conformational ensembles
        - Reducing steric clashes in predicted side-chain arrangements
        - Preparing protein structures for downstream structural analysis
        - Preparing protein models for docking and other structure-based
          computational workflows

        Notes
        -----
        Input structures are converted to PDB format before being processed by
        PIPPack. CIF files are converted using the Scipion CIF-to-PDB conversion
        utility, while other input formats are copied to the working directory
        with a PDB extension.

        The protocol can process a single structure or an ensemble in the same
        workflow. When multiple PDB files are generated, all resulting
        structures are retained in the output SetOfAtomStructs.

        GPU execution is optional. When enabled, the selected GPU IDs are passed
        to the PIPPack workflow.

        The resampling-specific parameters are only used when the resampling
        procedure is enabled.
    """
    _label = 'generate protein side chains'

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addHidden('useGpu', params.BooleanParam, default=True,
                       label="Use GPU for execution",
                       help="This protocol has both CPU and GPU implementation. Choose one.")

        form.addHidden('gpuList', params.StringParam, default='0',
                       label="Choose GPU IDs",
                       help="Comma-separated GPU devices that can be used.")

        form.addSection(label='Input')
        form.addParam('inputType', params.EnumParam,
                      choices=['AtomStruct', 'SetOfAtomStructs'], default=0,
                      label="Input format: ",
                      help='Select the input format.')

        form.addParam('inputStruct', params.PointerParam,
                      pointerClass='AtomStruct', condition='inputType==0',
                      label="Input structure: ",
                      help='Select the input structure.')

        form.addParam('inputEnsemble', params.PointerParam,
                      pointerClass='SetOfAtomStructs', condition='inputType==1',
                      label='Input ensemble: ',
                      help='Select a ensemble set (SetOfAtomStructs).')

        form.addParam('chiTemperature', params.FloatParam,
                    default=1.0,
                    label='Chi temperature: ',
                    help='Temperature used for sampling side-chain chi angles.'
        )

        form.addParam('recycles', params.IntParam,
                    default=3,
                    label='Recycles: ',
                    help='Number of recycling iterations during PIPPack inference.'
        )
        form.addParam('resampling', params.BooleanParam,
                      default=True,
                      label='Use resampling: ',
                      help=' Resampling procedure that is applied after PIPPack prediction to reduce the amount of clashing residues.'
                      )

        form.addParam(
            'sampleTemp',
            params.FloatParam,
            default=0.1, expertLevel=params.LEVEL_ADVANCED,
            condition='resampling',
            label='Sampling temperature: ',
            help='Temperature used when sampling alternative side-chain conformations.'
        )

        form.addParam(
            'clashOverlapTolerance',
            params.FloatParam,
            default=0.4, expertLevel=params.LEVEL_ADVANCED,
            condition='resampling',
            label='Clash overlap tolerance: ',
            help='Tolerance for atomic overlap when evaluating steric clashes.'
        )

        form.addParam(
            'proToleranceFactor',
            params.IntParam,
            default=12, expertLevel=params.LEVEL_ADVANCED,
            condition='resampling',
            label='Proline tolerance factor: ',
            help='Tolerance factor used for proline residues during clash evaluation.'
        )

        form.addParam(
            'maxIters',
            params.IntParam,
            default=50, expertLevel=params.LEVEL_ADVANCED,
            condition='resampling',
            label='Maximum iterations: ',
            help='Maximum number of resampling iterations.'
        )

        form.addParam(
            'metropolisTemp',
            params.FloatParam,
            default=0.000005, expertLevel=params.LEVEL_ADVANCED,
            condition='resampling',
            label='Metropolis temperature: ',
            help='Temperature controlling acceptance of candidate conformations.'
        )

        form.addParallelSection(threads=4, mpi=1)

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self._insertFunctionStep('prepareInputStep')
        self._insertFunctionStep('runPIPPack')
        self._insertFunctionStep('createOutputStep')

    def prepareInputStep(self):
        inputDir = self._getExtraPath('input')
        os.makedirs(inputDir, exist_ok=True)

        if self.inputType==1:
            for structure in self.inputEnsemble.get():
                inputFile = structure.getFileName()
                baseName = os.path.splitext(os.path.basename(inputFile))[0]
                outputFile = os.path.join(inputDir, f'{baseName}.pdb')

                self.convertOrCopy(inputFile, outputFile)
        else:
            inputFile = self.inputStruct.get().getFileName()
            baseName = os.path.splitext(os.path.basename(inputFile))[0]
            outputFile = os.path.join(inputDir, f'{baseName}.pdb')

            self.convertOrCopy(inputFile, outputFile)

    def runPIPPack(self):
        pluginDir = pwchemPlugin.getVar(PIPPACK_DIC['home'])
        pippackPath = os.path.join(pluginDir, 'PIPPack')

        scriptPath = os.path.join(
            os.path.dirname(__file__), "..", "scripts"
        )

        modelsPath = os.path.join(
            pippackPath,
            'model_weights'
        )

        resamplingFlag = (
            '--use_resampling'
            if self.resampling.get()
            else ''
        )

        args = (
            '--pippack_path "{}" '
            '--input_dir "{}" '
            '--output_dir "{}" '
            '--model_weights "{}" '
            '--chi_temperature {} '
            '--recycles {} '
            '{} '
            '--sample_temp {} '
            '--clash_overlap_tolerance {} '
            '--pro_tolerance_factor {} '
            '--max_iters {} '
            '--metropolis_temp {}'
        ).format(
            pippackPath,
            os.path.abspath(self._getExtraPath('input')),
            os.path.abspath(self._getExtraPath('output')),
            modelsPath,
            self.chiTemperature.get(),
            self.recycles.get(),
            resamplingFlag,
            self.sampleTemp.get(),
            self.clashOverlapTolerance.get(),
            self.proToleranceFactor.get(),
            self.maxIters.get(),
            self.metropolisTemp.get()
        )

        if self.useGpu.get():
            args += f' --gpu_ids {self.gpuList.get()}'

        pwchemPlugin.runScript(
            self,
            'pippack_workflow.py',
            args,
            env=PIPPACK_DIC,
            cwd=pippackPath,
            scriptDir=scriptPath
        )

    def createOutputStep(self):
        outputDir = self._getExtraPath('output')
        outputFiles = sorted(
            os.path.join(outputDir, f)
            for f in os.listdir(outputDir)
            if f.endswith('.pdb')
        )
        if not outputFiles:
            raise FileNotFoundError(
                f'No PDB files found in {outputDir}'
            )
        if len(outputFiles) == 1:
            outputStruct = AtomStruct()
            outputStruct.setFileName(outputFiles[0])

            self._defineOutputs(
                outputAtomStruct=outputStruct
            )
        else:
            outputSet = SetOfAtomStructs.create(self._getPath())
            for outputFile in outputFiles:
                outputStruct = AtomStruct()
                outputStruct.setFileName(outputFile)
                outputSet.append(outputStruct)

            self._defineOutputs(
                outputAtomStructs=outputSet
            )

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        summary = []
        return summary

    def _methods(self):
        methods = []
        return methods

    def _validate(self):
        validations = []
        return validations

    def _warnings(self):
        warnings = []
        return warnings

    # --------------------------- UTILS functions -----------------------------------
    def convertOrCopy(self, inModel, inpPDBModel):
        if inModel.endswith('.cif'):
            cifToPdb(inModel, inpPDBModel)
        else:
            shutil.copy(inModel, inpPDBModel)
