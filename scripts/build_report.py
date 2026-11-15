#!/usr/bin/env python3
"""Build the editable study-protocol report. Results are intentionally unpopulated."""
from __future__ import annotations

from pathlib import Path
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "Humanoid_Walking_Robustness_Report.docx"

NAVY = "17365D"
TEAL = "007C91"
PALE = "EAF2F8"
LIGHT = "F3F6F9"
MID = "526477"
WHITE = "FFFFFF"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=100, bottom=80, end=100) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_text(cell, text: str, *, bold=False, color=None, size=8.5) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Arial"
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths=None, font=8.5):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    for i, h in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], h, bold=True, color=WHITE, size=font)
        set_cell_shading(table.rows[0].cells[i], NAVY)
    set_repeat_table_header(table.rows[0])
    header_tr_pr = table.rows[0]._tr.get_or_add_trPr()
    header_tr_pr.append(OxmlElement("w:cantSplit"))
    for ri, row in enumerate(rows):
        new_row = table.add_row()
        row_pr = new_row._tr.get_or_add_trPr()
        row_pr.append(OxmlElement("w:cantSplit"))
        cells = new_row.cells
        for ci, value in enumerate(row):
            set_cell_text(cells[ci], value, size=font)
            if ri % 2 == 1:
                set_cell_shading(cells[ci], LIGHT)
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_para(doc, text="", style=None, bold_prefix=None, italic=False, color=None, after=5):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.1
    if bold_prefix and text.startswith(bold_prefix):
        a = p.add_run(bold_prefix)
        a.bold = True
        b = p.add_run(text[len(bold_prefix):])
        runs = [a, b]
    else:
        runs = [p.add_run(text)]
    for r in runs:
        r.font.name = "Arial"
        if italic:
            r.italic = True
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
    return p


def add_bullet(doc, text: str, level=0):
    style = "List Bullet" if level == 0 else "List Bullet 2"
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.25 + 0.2 * level)
    p.paragraph_format.first_line_indent = Inches(-0.12)
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(9.5)
    return p


def add_number(doc, text: str):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(9.5)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    for r in p.runs:
        r.font.name = "Arial"
        r.font.color.rgb = RGBColor.from_string(NAVY if level == 1 else TEAL)
    return p


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = paragraph.add_run("Page ")
    r.font.name = "Arial"
    r.font.size = Pt(8)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def set_keep_with_next(paragraph):
    paragraph.paragraph_format.keep_with_next = True


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.27)
    sec.page_height = Inches(11.69)
    sec.top_margin = Inches(0.72)
    sec.bottom_margin = Inches(0.72)
    sec.left_margin = Inches(0.78)
    sec.right_margin = Inches(0.78)
    sec.header_distance = Inches(0.3)
    sec.footer_distance = Inches(0.32)

    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    styles["Normal"].font.size = Pt(9.5)
    styles["Normal"].font.color.rgb = RGBColor(35, 46, 58)
    for sname, size, color in [("Title", 27, NAVY), ("Subtitle", 13, TEAL), ("Heading 1", 16, NAVY), ("Heading 2", 12, TEAL), ("Heading 3", 10, MID)]:
        st = styles[sname]
        st.font.name = "Arial"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        st.font.size = Pt(size)
        st.font.bold = sname != "Subtitle"
        st.font.color.rgb = RGBColor.from_string(color)

    header = sec.header.paragraphs[0]
    header.text = "HUMANOID WALKING ROBUSTNESS | ISAAC LAB + ISAAC SIM"
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in header.runs:
        r.font.name = "Arial"
        r.font.bold = True
        r.font.size = Pt(7.5)
        r.font.color.rgb = RGBColor.from_string(TEAL)
    footer = sec.footer.paragraphs[0]
    footer.add_run("M.Tech. Robotics | IIT Jodhpur | Preliminary protocol report    ")
    for r in footer.runs:
        r.font.name = "Arial"
        r.font.size = Pt(7.5)
        r.font.color.rgb = RGBColor.from_string(MID)
    add_page_number(footer)

    # Cover page
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(78)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("ROBOTICS PROJECT REPORT")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor.from_string(TEAL)

    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(12)
    p.add_run("Humanoid Walking Robustness\nunder Domain Randomization")

    p = doc.add_paragraph(style="Subtitle")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("A controlled PPO study using the Unitree G1 in Isaac Lab 3.0")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(34)
    r = p.add_run("Rikraj  |  Kanika Chouhan")
    r.bold = True
    r.font.name = "Arial"
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor.from_string(NAVY)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("M.Tech. Robotics\nIndian Institute of Technology Jodhpur\nRajasthan, India")
    r.font.name = "Arial"
    r.font.size = Pt(11)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(46)
    r = p.add_run("PLATFORM: NVIDIA BREV | ISAAC LAB 3.0.0 | ISAAC SIM 6.2.0\nPPO: RSL-RL | PHYSICS: NEWTON MUJOCO-WARP")
    r.font.name = "Arial"
    r.font.bold = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor.from_string(TEAL)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(32)
    r = p.add_run("14 November 2026")
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor.from_string(MID)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(27)
    r = p.add_run("STATUS: STUDY PROTOCOL AND IMPLEMENTATION PLAN - EXPERIMENTAL RESULTS PENDING")
    r.bold = True
    r.font.name = "Arial"
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor.from_string("A64B00")

    doc.add_page_break()

    add_heading(doc, "Abstract", 1)
    add_para(doc, "This project investigates whether domain randomization improves the robustness of reinforcement-learning-based humanoid walking. Two Proximal Policy Optimization (PPO) controllers are trained using the same Unitree G1 flat-ground velocity-tracking task in Isaac Lab: a fixed-condition baseline and a domain-randomized controller trained with varying foot-contact friction and bounded horizontal disturbances. Training uses the RSL-RL PPO implementation and the Newton MuJoCo-Warp physics backend on an NVIDIA Brev GPU instance. The two controllers are evaluated on an identical factorial test grid containing nominal, in-range, and held-out friction coefficients, together with several disturbance strengths. Evaluation metrics are episode success rate, mean planar velocity-tracking error, fall frequency, and survival time. Three independent training seeds are planned per condition. The report specifies the simulator setup, randomization implementation, training protocol, evaluation adapter, analysis pipeline, and required deliverables. No numerical performance outcome is claimed in this protocol-stage version; the result tables and plots must be populated only from completed Brev evaluation CSV files.")

    add_heading(doc, "1. Introduction", 1)
    add_para(doc, "Humanoid robots operate through a high-dimensional, underactuated locomotion system. Small changes in ground contact, friction, or external disturbances can alter the relationship between a joint command and the resulting body motion. A policy trained only under one fixed simulation condition may therefore perform well in its training environment but degrade when the contact dynamics or disturbances change.")
    add_para(doc, "Reinforcement learning provides a practical way to learn command-conditioned walking policies by collecting many simulated rollouts. Domain randomization attempts to expose a policy to a distribution of plausible environment conditions during training, encouraging it to perform across a broader set of dynamics. Prior humanoid-locomotion studies have used randomized simulation environments and disturbance testing, including work on sim-to-real humanoid walking [6, 7]. This project focuses on a smaller, controlled empirical question and does not propose a new RL algorithm.")
    add_heading(doc, "1.1 Central research question", 2)
    add_para(doc, "Does training with friction variation and random horizontal pushes improve Unitree G1 walking robustness compared with training under fixed contact conditions, when both PPO controllers are tested on the same nominal and held-out disturbance conditions?")
    add_heading(doc, "1.2 Objectives", 2)
    for t in [
        "Reproduce a feed-forward PPO controller on the existing Isaac Lab Unitree G1 flat-ground velocity task.",
        "Create a matched DR controller by randomizing foot-contact friction and applying bounded interval disturbances while keeping the PPO architecture, reward configuration, observation/action interfaces, and training budget unchanged.",
        "Evaluate both policies over the same friction-disturbance matrix using three training seeds and report success, velocity error, falls, and survival time.",
    ]:
        add_bullet(doc, t)
    add_heading(doc, "1.3 Scope", 2)
    add_para(doc, "The study is limited to flat-ground, command-conditioned walking in simulation. It excludes robot hardware, sim-to-real deployment, stairs, jumping, manipulation, a new robot model, and the design of a new RL algorithm. The results will support a bounded simulation robustness comparison only; they will not establish physical-robot safety.")

    add_heading(doc, "2. Related Work", 1)
    add_para(doc, "Humanoid-Gym presents an Isaac Gym-based framework for humanoid locomotion that incorporates dynamics randomization and sim-to-sim validation, and reports transfer experiments on humanoid platforms [6]. Radosavovic et al. demonstrate real-world humanoid locomotion using a policy trained with large-scale reinforcement learning over randomized simulated environments; the study also evaluates robustness to external disturbances [7]. These papers motivate the use of randomized simulation conditions, but they differ from the present project in robot platform, model architecture, scale, and hardware scope. The present experiment is intentionally narrower: it uses an existing G1 task and tests the combined effect of friction randomization plus bounded pushes against a fixed-condition baseline.")

    add_heading(doc, "3. Platform and Simulation Environment", 1)
    add_heading(doc, "3.1 Compute platform: NVIDIA Brev", 2)
    add_para(doc, "The experiment is designed to run on an NVIDIA Brev VM-mode GPU instance. NVIDIA's current Brev Isaac Sim deployment guide recommends a single NVIDIA L40S GPU as a starting configuration [5]. The actual GPU model, driver, VRAM, CPU, RAM, operating-system image, and instance identifier must be recorded from the deployed VM. The Brev web-streaming ports (49100, 47998, and 8210) should be exposed only to the team's client IP when streaming is required [5]. Training and numerical evaluation can run headless, so streaming should remain disabled during long training jobs unless needed for monitoring.")
    add_heading(doc, "3.2 Isaac Lab and Isaac Sim versions", 2)
    add_para(doc, "The implementation targets the Isaac Lab 3.0.0 release branch, Python 3.12, and the registered task `Isaac-Velocity-Flat-G1` [1, 3]. Isaac Lab 3.0 supports a kit-less Newton workflow and treats Isaac Sim as an optional integration; Isaac Sim 6.2 is the supported full Kit/PhysX integration for this release [1]. For reproducibility, the main training and quantitative evaluation runs use the same `newton_mjwarp` backend. Isaac Sim/Kit is used for interactive visualization and demonstration capture where supported. Changing the physics backend between training and testing would add a confounding variable and is therefore not part of the main experiment [2].")
    add_heading(doc, "3.3 Robot task", 2)
    add_para(doc, "The selected environment is the existing Unitree G1 flat-ground velocity-tracking task. It provides a robot articulation, joint-position action interface, proprioceptive observations, velocity commands, locomotion rewards, and termination conditions. The upstream flat-task command ranges are retained and must be verified against the exact release checkout: forward linear velocity from 0 to 1.0 m/s, lateral linear velocity from -0.5 to 0.5 m/s, and yaw angular velocity from -1.0 to 1.0 rad/s [3]. The G1 flat configuration selects the Newton MuJoCo-Warp solver in the target release [3].")

    add_heading(doc, "4. Experimental Methodology", 1)
    add_heading(doc, "4.1 Controlled comparison", 2)
    add_para(doc, "Two PPO controllers are trained from the same task and registered RSL-RL agent configuration. Only the specified contact-friction and push events differ between the baseline and DR training groups. The following factors remain identical: robot model, flat terrain, observations, actions, reward terms and weights, PPO network and hyperparameters, command sampler, simulation time step, decimation, number of parallel environments, number of iterations, checkpoint-selection rule, and training seeds.")
    add_table(doc,
        ["Parameter", "Baseline", "Domain-randomized"],
        [
            ["Foot-contact static friction", "Fixed at 1.0", "Sampled from [0.5, 1.25] at startup"],
            ["Foot-contact dynamic friction", "Fixed at 1.0", "Configured as [0.5, 1.25]; Newton uses one friction coefficient"],
            ["Horizontal push event", "Disabled", "Enabled every 5.0 s"],
            ["x/y root velocity perturbation", "None", "Each axis sampled from [-0.5, 0.5] m/s"],
            ["Base mass and base CoM randomization", "Disabled", "Disabled"],
            ["PPO training budget", "1500 iterations", "1500 iterations"],
            ["Parallel environments", "64", "64"],
            ["Training seeds", "11, 22, 33", "11, 22, 33"],
            ["Physics backend", "newton_mjwarp", "newton_mjwarp"],
        ], widths=[2.05, 2.25, 3.0], font=8.2)

    doc.add_page_break()
    add_heading(doc, "4.2 Friction randomization implementation", 2)
    add_para(doc, "The G1 velocity task inherits a startup `physics_material` event that uses Isaac Lab's `randomize_rigid_body_material` function. The project patch narrows this event's asset selector to the ankle-roll/foot-contact geometry pattern `.*_ankle_roll_link`. For the baseline, the friction ranges are fixed at 1.0. For DR training, the static-friction range is [0.5, 1.25]. The terrain material remains at its task default. Because the contact material combination is multiplicative in the target task, changing the foot material changes the effective foot-ground contact coefficient while leaving the plane material fixed. This should be described accurately as foot-contact material randomization rather than claiming that the plane mesh itself is resampled.")
    add_para(doc, "The Isaac Lab 3.0 MDP API documents that material randomization supports backend-specific semantics. For Newton, the operative friction is a single coefficient; dynamic friction and PhysX material buckets should not be interpreted as independently active random variables [4]. The implementation records this backend limitation as part of the method.")

    add_heading(doc, "4.3 Push disturbance implementation", 2)
    add_para(doc, "The DR configuration retains the task's interval `push_robot` event and sets the interval to 5.0 seconds. The event calls `push_by_setting_velocity`, which changes the robot root velocity to a random value within the provided x/y ranges [4]. In this experiment, the maximum absolute x/y velocity perturbation is 0.5 m/s per axis during training. This is a bounded velocity-reset disturbance proxy. It is not a calibrated external force in Newtons and should not be labelled as a measured force impulse.")

    add_heading(doc, "4.4 Training protocol", 2)
    for t in [
        "Verify that the registered G1 flat task resets and steps using the random-agent smoke test.",
        "Run a 10-iteration, 16-environment PPO smoke test for baseline and DR modes.",
        "Train baseline seeds 11, 22, and 33 for 1500 iterations each.",
        "Train DR seeds 11, 22, and 33 for 1500 iterations each.",
        "Save the final checkpoint and retain any best validation checkpoint using training/validation data only; do not choose a checkpoint by inspecting test metrics.",
        "Record source commit, package versions, GPU/driver details, exact run arguments, and checkpoint paths.",
    ]:
        add_number(doc, t)
    add_para(doc, "The selected task uses a 0.005 s physics step and decimation 4 (0.02 s control step), with a nominal 20 s episode horizon in the shared velocity configuration. These values must be verified from the exact release configuration used in the run and copied into the experiment metadata.")

    add_heading(doc, "5. Evaluation Protocol", 1)
    add_heading(doc, "5.1 Factorial test matrix", 2)
    add_table(doc,
        ["Evaluation factor", "Values", "Purpose"],
        [
            ["Contact friction coefficient", "0.4, 0.5, 0.875, 1.0, 1.25, 1.4", "0.4 and 1.4 are held-out values outside DR training range; others are in-range/nominal"],
            ["Push velocity limit", "0.0, 0.25, 0.5 m/s per x/y axis", "Nominal, moderate, and training-maximum disturbance"],
            ["Episodes per condition", "100", "Estimate condition-level episode metrics"],
            ["Evaluation seed", "2027 (paired across policies)", "Keep command/push randomness reproducible"],
        ], widths=[1.7, 2.25, 3.35], font=8.1)
    add_para(doc, "The 6 x 3 grid contains 18 test conditions for each checkpoint. Across two policies, three training seeds, and 100 episodes per condition, the complete matrix contains 10,800 episode records. The two held-out friction endpoints test extrapolation beyond the DR friction training interval. The 0.25 and 0.5 m/s push tests measure performance under nonzero disturbances; zero push is the nominal disturbance control.")

    doc.add_page_break()
    add_heading(doc, "5.2 Metrics", 2)
    add_table(doc,
        ["Metric", "Operational definition", "Preferred direction"],
        [
            ["Mean planar velocity error", "Episode time-average of ||v_xy_actual - v_xy_command||2, in m/s", "Lower"],
            ["Fall frequency", "Fraction of episodes ending in non-timeout termination", "Lower"],
            ["Survival time", "Control steps before termination x physics_dt x decimation, in seconds", "Higher"],
            ["Episode success rate", "No fall, at least 95% of episode horizon completed, and mean velocity error <= 0.35 m/s", "Higher"],
        ], widths=[1.45, 4.55, 1.1], font=8.1)
    add_para(doc, "The 0.35 m/s velocity-error threshold and 95% episode-duration criterion are project-defined success criteria. They are not simulator defaults, clinical/industrial safety limits, or formal stability guarantees. The evaluation code stores the mean error under both `mean_planar_velocity_error_m_s` and the compatibility field `mean_abs_velocity_error`.")

    add_heading(doc, "5.3 Statistical analysis", 2)
    add_para(doc, "For each policy and test condition, compute the episode-level success rate, fall frequency, mean velocity error, and mean survival time. Also report the three training-seed results separately. The training seed is the independent replicate for assessing run-to-run variability; the 100 test episodes from one checkpoint are not 100 independent training replications. Report the mean and standard deviation across the three training seeds and the paired difference between randomized and baseline models at matching seed index and matching test condition. Avoid conclusions based only on pooled episode counts.")

    add_heading(doc, "6. Implementation and Reproducibility", 1)
    add_para(doc, "The accompanying source package contains the environment patcher, training scripts, evaluation collector, analysis script, configuration file, and Brev setup instructions. It is organized to address the raw-data issue encountered in the earlier plotting attempt: evaluation must first generate episode rows, and the analysis stage must reject empty or incomplete per-checkpoint evaluation matrices.")
    add_table(doc,
        ["File", "Role"],
        [
            ["config/experiment.yaml", "Canonical experiment settings and metric definitions"],
            ["scripts/patch_g1_task.py", "Idempotently patches the release G1 flat config and creates a backup"],
            ["scripts/run_smoke_test.sh", "Checks task registration and both config modes before full runs"],
            ["scripts/run_experiments.sh", "Executes the six PPO training runs"],
            ["scripts/evaluate.py", "Loads a checkpoint and collects 100 episodes for every friction/push condition"],
            ["scripts/run_evaluation_matrix.sh", "Runs evaluation for all six checkpoints"],
            ["scripts/analyze_results.py", "Aggregates CSVs and generates report-ready plots and comparison tables"],
            ["scripts/play_policy.sh", "Loads a policy in Isaac Lab and records a short demonstration video"],
            ["README.md", "Step-by-step Brev setup, commands, deliverables, and troubleshooting"],
        ], widths=[2.55, 4.55], font=8.0)

    add_heading(doc, "6.1 Run order", 2)
    run_order = [
        "Provision Brev GPU VM and record hardware/software metadata.",
        "Install the Isaac Lab 3.0.0 release and verify the G1 flat task ID.",
        "Place the project beside the Isaac Lab checkout; apply and review the idempotent G1 task patch.",
        "Run the smoke test for both conditions.",
        "Train and record all six PPO checkpoints.",
        "Evaluate all checkpoints over the identical 18-condition test matrix.",
        "Verify each per-checkpoint CSV contains 1800 episode rows plus one header.",
        "Run analysis to generate summary tables and the four evaluation plots.",
        "Capture baseline and DR playback videos in Isaac Sim/Kit where supported.",
        "Populate the results tables, discussion, and conclusion from the generated data; archive all source and metadata.",
    ]
    for i, t in enumerate(run_order, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(f"{i}.  {t}")
        r.font.name = "Arial"
        r.font.size = Pt(9.5)

    add_heading(doc, "7. Project Timeline and Contributions", 1)
    add_table(doc,
        ["Week", "Activity", "Rikraj", "Kanika Chouhan"],
        [
            ["1", "Review, setup, baseline verification", "Literature review, Brev setup, environment baseline verification", "PPO workflow familiarization and baseline execution"],
            ["2", "Baseline training and DR implementation", "Environment setup, metric definitions, baseline training support", "Friction variation and push-event implementation"],
            ["3", "DR training and matched testing", "Modified-policy training and testing", "Randomized-policy training; matched scenarios and data collection"],
            ["4", "Analysis, demonstration, reporting", "Result interpretation and code documentation", "Plots, demonstration videos, and report preparation"],
        ], widths=[0.55, 1.55, 2.5, 2.5], font=7.7)
    add_para(doc, "The four-week schedule is a project plan; actual completion dates, compute delays, and task ownership changes should be entered in the final submission log.")

    add_heading(doc, "8. Required Deliverables", 1)
    for t in [
        "Source code, task-configuration patch, YAML experiment configuration, and README instructions.",
        "Six trained policy checkpoints (baseline and DR, each with training seeds 11, 22, and 33).",
        "Raw per-episode evaluation CSV data with all test conditions and policy/seed identifiers.",
        "Processed metric tables, paired seed differences, and four plots: success vs friction, tracking error vs push, fall frequency vs push, and the success heatmap.",
        "Two or more demonstration videos with policy, seed, friction, and push metadata.",
        "Reproducibility metadata: Isaac Lab/Sim versions, Git commit, Python/driver/GPU details, commands, checkpoint manifest, and random seeds.",
        "Completed final report with actual results and limitations; no placeholder values should remain in the submitted version.",
    ]:
        add_bullet(doc, t)

    add_heading(doc, "9. Results and Discussion - To Be Populated from Brev Runs", 1)
    add_para(doc, "No simulator training or evaluation output was available at report-generation time. Accordingly, all numerical result cells remain pending. Do not replace these cells with expected or illustrative values. Populate them directly from `results/processed/metrics_by_training_seed.csv` and `metrics_summary.csv` after verifying that the raw CSVs contain episode rows.", italic=True, color="A64B00")
    add_heading(doc, "9.1 Results table", 2)
    add_table(doc,
        ["Test condition", "Policy", "Success rate", "Mean error (m/s)", "Fall frequency", "Survival (s)"],
        [
            ["Nominal: friction 1.0, push 0.0", "Baseline / DR", "Pending", "Pending", "Pending", "Pending"],
            ["Low held-out: friction 0.4, push 0.0", "Baseline / DR", "Pending", "Pending", "Pending", "Pending"],
            ["High held-out: friction 1.4, push 0.0", "Baseline / DR", "Pending", "Pending", "Pending", "Pending"],
            ["Nominal friction 1.0, push 0.25", "Baseline / DR", "Pending", "Pending", "Pending", "Pending"],
            ["Nominal friction 1.0, push 0.5", "Baseline / DR", "Pending", "Pending", "Pending", "Pending"],
        ], widths=[2.0, 0.85, 1.0, 1.1, 1.0, 1.05], font=7.1)
    add_heading(doc, "9.2 Required figures", 2)
    add_table(doc,
        ["Figure", "Question answered", "File generated by analysis"],
        [
            ["Success vs friction", "Does either policy retain success across low, nominal, and high contact friction?", "success_vs_friction.png"],
            ["Velocity error vs push", "How does tracking quality change as horizontal perturbation increases?", "velocity_error_vs_push.png"],
            ["Fall frequency vs push", "Does the probability of early termination rise under stronger pushes?", "fall_frequency_vs_push.png"],
            ["Success heatmap", "Which combinations of friction and push are most difficult for each policy?", "success_heatmap_by_policy.png"],
        ], widths=[1.5, 3.65, 2.2], font=7.7)
    add_heading(doc, "9.3 Discussion prompts", 2)
    for t in [
        "Compare baseline and DR policies at nominal friction and no pushes. Did randomization trade nominal performance for robustness, or preserve both?",
        "Compare success and fall frequency at held-out friction 0.4 and 1.4. Are gains symmetric, or does the policy fail differently at low and high friction?",
        "Compare tracking error and falls at push strengths 0.25 and 0.5 m/s. Does DR improve recovery without making normal tracking worse?",
        "Report training-seed spread. Are observed improvements consistent across all three paired seeds, or driven by one run?",
        "Discuss confounds and backend limitations, especially that the Newton contact model uses one friction coefficient and the push mechanism is a root-velocity perturbation.",
    ]:
        add_bullet(doc, t)

    add_heading(doc, "10. Risks, Mitigations, and Limitations", 1)
    add_table(doc,
        ["Risk / limitation", "Mitigation / reporting requirement"],
        [
            ["Brev GPU memory or throughput constraints", "Smoke-test first; reduce environment count for both groups equally; record the actual value."],
            ["Isaac Lab API changes across releases", "Pin release/3.0.0; use the supplied patcher; verify source and task registration."],
            ["Empty or incomplete evaluation CSV", "Require 1800 rows per checkpoint file; abort analysis if no episode data exists."],
            ["Newton friction semantics differ from PhysX", "Use one backend for all numeric runs and disclose the single-coefficient limitation."],
            ["Pushes are velocity resets rather than force impulses", "Describe them as root-velocity disturbances; do not report a force magnitude in N."],
            ["Friction and pushes are varied together", "Interpret results as the combined DR package; separate ablation studies would be needed to isolate each factor."],
            ["Only three training seeds", "Report all seed-level values and avoid strong claims from pooled episodes."],
            ["Simulation-only findings", "Make no claim of physical-robot safety or sim-to-real transfer."],
        ], widths=[2.25, 4.85], font=7.8)

    add_heading(doc, "11. Conclusion", 1)
    add_para(doc, "This project defines a reproducible comparison of fixed-condition and domain-randomized PPO controllers for Unitree G1 flat-ground walking. It uses Isaac Lab 3.0.0, the registered G1 velocity task, the RSL-RL PPO runner, and a consistent Newton MuJoCo-Warp simulation backend on NVIDIA Brev. The experiment varies foot-contact friction and bounded horizontal root-velocity disturbances during DR training, then tests both policies on an identical nominal and held-out condition grid. The final conclusion must be written after the six training runs and complete evaluation matrix are available. Until then, the research question remains open and no improvement claim is justified.")

    add_heading(doc, "References", 1)
    refs = [
        "[1] NVIDIA. Isaac Lab 3.0 Installation. https://docs.nvidia.com/isaaclab/3.0/isaac-lab/installation",
        "[2] NVIDIA. Isaac Lab 3.0 Reinforcement Learning. https://docs.nvidia.com/isaaclab/3.0/concepts/reinforcement-learning",
        "[3] NVIDIA / Isaac Lab Project. G1 Flat Environment Configuration, release/3.0.0. https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/flat_env_cfg.py",
        "[4] NVIDIA / Isaac Lab Project. Isaac Lab 3.0 MDP API: randomize_rigid_body_material and push_by_setting_velocity. https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab/isaaclab.envs.mdp.html",
        "[5] NVIDIA. NVIDIA Brev Deployment - Isaac Sim Documentation. https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_advanced_cloud_setup_brev.html",
        "[6] Gu, X., Wang, Y.-J., & Chen, J. (2024). Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer. arXiv:2404.05695. https://arxiv.org/abs/2404.05695",
        "[7] Radosavovic, I., Xiao, T., Zhang, B., Darrell, T., Malik, J., & Sreenath, K. (2024). Real-world humanoid locomotion with reinforcement learning. Science Robotics, 9, eadi9579. https://doi.org/10.1126/scirobotics.adi9579",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        r = p.add_run(ref)
        r.font.name = "Arial"
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor.from_string(MID)

    # Document metadata
    doc.core_properties.title = "Humanoid Walking Robustness under Domain Randomization"
    doc.core_properties.subject = "Isaac Lab 3.0 PPO robustness study protocol"
    doc.core_properties.author = "Rikraj; Kanika Chouhan"
    doc.core_properties.keywords = "Unitree G1, Isaac Lab, Isaac Sim, NVIDIA Brev, PPO, domain randomization"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
