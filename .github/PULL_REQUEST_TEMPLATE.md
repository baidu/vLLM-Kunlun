## PR Description

<!--
Please provide a clear and concise description of this PR:
- What problem does it solve?
- Why is this change needed?
- What is the overall approach?
-->

FIX #xxxx  
<!-- Link the existing issue(s) this PR resolves, e.g.:
FIX #1234
-->

---

## Checklist (Required)

Before submitting this PR, please ensure that all the following items are completed:

- [ ] All code changes pass the [`pre-commit`](https://github.com/baidu/vLLM-Kunlun/blob/main/CONTRIBUTING.md) checks.
- [ ] Commits are signed off using `git commit -s`.
- [ ] The PR title is properly classified (see below).

---

## PR Type

Please prefix the PR title with one or more of the following labels to help reviewers quickly understand the nature of the change:

- `[Feature]` – New features or enhancements (e.g. Attention, Communicator, Kernel, Worker, etc.)
- `[Bugfix]` – Bug fixes
- `[CI/Build]` – CI, build system, or infrastructure improvements
- `[Doc]` – Documentation updates or fixes
- `[Misc]` – Other changes that do not fit the above categories (use sparingly)

> **Note:** If the PR spans multiple categories, include all relevant prefixes.

---

<details>
<summary><b>Detailed Checklist (Click to Expand)</b></summary>

<p>Thank you for contributing to <b>vLLM Kunlun</b>!  
To help us maintain high code quality and streamline the review process, please ensure your PR meets the following requirements.</p>

<h3>1. Code Quality</h3>

<ul>
    <li>All linting and formatting checks pass (<code>pre-commit</code>).</li>
    <li>The code is well-structured and sufficiently documented.</li>
    <li>The change is designed with maintainability and readability in mind.</li>
</ul>

<h3>2. Testing</h3>

<ul>
    <li>Relevant unit tests are added or updated.</li>
    <li>Integration tests are included when applicable.</li>
    <li>Existing tests continue to pass.</li>
</ul>

<h3>3. DCO Compliance</h3>

<p>This project follows the
<a href="https://github.com/vllm-project/vllm/blob/main/DCO">Developer Certificate of Origin (DCO)</a>.</p>

<ul>
    <li>All commits include a <code>Signed-off-by:</code> line.</li>
    <li>Use <code>git commit -s</code> to automatically add the sign-off.</li>
</ul>

<h3>4. Review Expectations</h3>

<p>During the review process, maintainers may:</p>

<ul>
    <li>Request code refactoring or additional tests.</li>
    <li>Ask for clarifications on design decisions.</li>
    <li>Suggest performance, stability, or maintainability improvements.</li>
</ul>

<p>We appreciate your patience and collaboration throughout the review process!</p>

</details>
