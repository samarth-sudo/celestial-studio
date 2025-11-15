# System Issues Tracking

**Last Updated**: 2025-01-14
**Total Issues**: 23
**Status**: Fixes In Progress

---

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| 🔴 Critical | 5 | 0 fixed, 5 pending |
| 🟠 High Priority | 4 | 0 fixed, 4 pending |
| 🟡 Medium Priority | 7 | 0 fixed, 7 pending |
| 🟢 Low Priority | 7 | 0 fixed, 7 pending |

---

## 🔴 CRITICAL ISSUES

### 1. Missing Algorithm Compilation Error Handling
**File**: `frontend/src/services/AlgorithmManager.ts:269-270`
**Severity**: CRITICAL
**Status**: ❌ Pending

**Issue**: When `executePathPlanning()` is called with non-compiled algorithm, throws error and prevents custom algorithms from executing.

**Code**:
```typescript
if (!algorithm.compiledFunction) {
  throw new Error(`Algorithm ${algorithmId} is not compiled (display only)`)
}
```

**Fix**: Auto-compile on first use instead of throwing error.

---

### 2. Simulator Falls Back to Built-in A* Silently
**File**: `frontend/src/components/Simulator.tsx:220-222`
**Severity**: CRITICAL
**Status**: ❌ Pending

**Issue**: Custom algorithm execution fails but recovery is silent, leaving user confused why their algorithm isn't being used.

**Fix**: Log clearly which algorithm is being used and why.

---

### 3. eval() Security Vulnerability
**File**: `frontend/src/services/AlgorithmManager.ts:362`
**Severity**: CRITICAL (Security)
**Status**: ❌ Pending

**Issue**: Using `eval()` inside Function constructor allows arbitrary code execution if algorithm generation produces malicious code.

**Code**:
```typescript
if (typeof eval(functionName) === 'function') {
  return eval(functionName)(...args)
}
```

**Fix**: Replace with direct property access: `compiledFunction[functionName](...args)`

---

### 4. Chat Error - Generic Exception Handler
**File**: `backend/api/conversational_chat.py:857-865`
**Severity**: CRITICAL
**Status**: ❌ Pending

**Issue**: Generic "Sorry, there was an error" hides actual errors from users.

**Root Causes**:
- Ollama connection failure
- Timeout (30 seconds may be too short)
- JSON parsing errors

**Fix**: User-friendly error messages with specific troubleshooting steps.

---

### 5. Missing Import Error - Robot API
**File**: `backend/main.py:43-48`
**Severity**: CRITICAL
**Status**: ❌ Pending

**Issue**: Robot API router conditionally disabled but commented out - causes confusion and potential 404 errors.

**Code**:
```python
# Temporarily disabled for startup
# try:
#     from api.robot_api import router as robot_router
# except ImportError:
#     from backend.api.robot_api import router as robot_router
robot_router = None
```

**Fix**: Either enable or remove completely with proper documentation.

---

## 🟠 HIGH PRIORITY ISSUES

### 6. No Ollama Health Check
**Location**: `backend/api/conversational_chat.py`
**Severity**: HIGH
**Status**: ❌ Pending

**Issue**: No validation that Ollama is running before accepting chat requests.

**Impact**: Users get cryptic errors instead of "Please start Ollama"

**Fix**: Add health check endpoint and validate before processing.

---

### 7. Hardcoded Backend URL
**Files**:
- `frontend/src/services/AlgorithmManager.ts:46`
- `frontend/src/components/ConversationalChat.tsx:93, 183`

**Severity**: HIGH
**Status**: ❌ Pending

**Issue**: No environment variable support - `http://localhost:8000` hardcoded.

**Impact**: Cannot deploy frontend without code changes.

**Fix**: Use `import.meta.env.VITE_BACKEND_URL`

---

### 8. Scene Config Adaptation Type Mismatch
**File**: `frontend/src/components/ConversationalChat.tsx:62-67`
**Severity**: HIGH
**Status**: ❌ Pending

**Issue**: Robot config merges backend fields but type checking is loose, causing potential type mismatches.

**Fix**: Strict type validation and transformation layer.

---

### 9. Isaac Lab Module Import Failures
**File**: `backend/main.py:1341-1342`
**Severity**: HIGH
**Status**: ❌ Pending

**Issue**: Isaac Lab import failures don't prevent startup, causing runtime 503 errors.

**Code**:
```python
except ImportError:
    ISAAC_LAB_AVAILABLE = False
    print("⚠️  Isaac Lab module not available - Modal simulation disabled")
```

**Fix**: Disable Isaac Lab features in UI if not available.

---

## 🟡 MEDIUM PRIORITY ISSUES

### 10. No Error Boundary on Simulator
**File**: `frontend/src/components/Simulator.tsx`
**Severity**: MEDIUM
**Status**: ❌ Pending

**Issue**: Three.js or Rapier errors crash entire simulator without graceful fallback.

**Impact**: White screen of death.

**Fix**: Wrap in React Error Boundary.

---

### 11. Path Planning Can Have Zero-Length Path
**File**: `frontend/src/components/Simulator.tsx:245-253`
**Severity**: MEDIUM
**Status**: ❌ Pending

**Issue**: When waypoints array is empty, path stats panel shows undefined behavior.

**Fix**: Validate path before setting state, show clear "No path found" message.

---

### 12. Multiple Fetch Calls May Block
**File**: `backend/main.py:1750-1820`
**Severity**: MEDIUM
**Status**: ❌ Pending

**Issue**: `run_training_with_progress()` blocks entire event loop during Modal training.

**Impact**: Other requests may timeout.

**Fix**: Use async/await properly or background tasks.

---

### 13. Missing CORS Headers for WebRTC
**File**: `backend/main.py:102`
**Severity**: MEDIUM
**Status**: ❌ Pending

**Issue**: CORS allows all origins in dev (`["*"]`) but WebRTC needs specific headers.

**Fix**: Add proper WebRTC CORS headers.

---

### 14. No Loading State on Algorithm Generator
**File**: `frontend/src/components/AlgorithmControls.tsx:84-85`
**Severity**: MEDIUM
**Status**: ❌ Pending

**Issue**: Alert says "takes 10-20 seconds" but no visual progress indicator.

**Impact**: User thinks app is frozen.

**Fix**: Add spinner and progress messages.

---

### 15. Algorithm Code Doesn't Show Compilation Errors
**File**: `frontend/src/components/AlgorithmControls.tsx`
**Severity**: MEDIUM
**Status**: ❌ Pending

**Issue**: Compilation errors only logged to console, not shown to user.

**Fix**: Display compilation errors in UI with fix suggestions.

---

### 16. Path Visualizer May Not Update
**File**: `frontend/src/components/Simulator.tsx:259`
**Severity**: MEDIUM
**Status**: ❌ Pending

**Issue**: Path only recalculates when `activeAlgorithmId` changes, not when obstacles update.

**Fix**: Add `editableObjects` to dependency array.

---

## 🟢 LOW PRIORITY / UI ISSUES

### 17. Custom Algorithm → Simulator Bridge Incomplete
**File**: `frontend/src/components/Simulator.tsx:206-223`
**Severity**: LOW
**Status**: ❌ Pending

**Issue**: Path execution shows algorithm used but robot doesn't actually follow custom algorithm paths consistently.

**Fix**: Verify algorithm execution flow end-to-end.

---

### 18. Multi-Robot Scene Generation Not Integrated
**File**: `backend/api/conversational_chat.py:787`
**Severity**: LOW
**Status**: ❌ Pending

**Issue**: Multi-robot detection returns message but doesn't actually add robots.

**Fix**: Implement multi-robot scene generation or remove feature.

---

### 19. Isaac Lab Scene Converter Not Validated
**File**: `backend/main.py:1371`
**Severity**: LOW
**Status**: ❌ Pending

**Issue**: Scene conversion result not checked before passing to Modal.

**Fix**: Validate conversion output format.

---

### 20. Export Package Generation No Error Recovery
**File**: `backend/api/conversational_chat.py:195-267`
**Severity**: LOW
**Status**: ❌ Pending

**Issue**: If PackageGenerator fails, error message is generic.

**Fix**: Specific error messages for each failure mode.

---

### 21. URDF Parser Doesn't Validate Input
**File**: `backend/main.py:574-621`
**Severity**: LOW
**Status**: ❌ Pending

**Issue**: No validation that uploaded URDF is valid XML before parsing.

**Impact**: Crashes on malformed files instead of user-friendly error.

**Fix**: XML validation before parsing.

---

### 22. Conversational Chat Doesn't Initialize Algorithms List
**File**: `frontend/src/components/ConversationalChat.tsx:813-814`
**Severity**: LOW
**Status**: ❌ Pending

**Issue**: Algorithms list only initialized if needed for export.

**Impact**: Export feature may fail.

**Fix**: Initialize on simulation creation.

---

### 23. No Timeout Increase for Complex Extractions
**File**: `backend/context_manager.py:54`
**Severity**: LOW
**Status**: ❌ Pending

**Issue**: 30-second timeout may be too short for complex scene descriptions.

**Fix**: Increase to 60 seconds.

---

## Root Cause Analysis

### Configuration Issues (8)
- Hardcoded URLs
- No environment variables
- Missing health checks
- Improper CORS configuration

### Error Handling (9)
- Silent failures
- Generic error messages
- No validation
- Missing try-catch blocks

### Security (2)
- eval() usage
- No input validation

### Type Safety (2)
- Loose type checking
- Backend/frontend mismatch

### Async/Concurrency (2)
- Blocking operations
- Event loop issues

---

## Fix Priority Order

### Week 1 (Critical + High)
1. ✅ Fix chat error messages
2. ✅ Add Ollama health check
3. ✅ Remove eval() security issue
4. ✅ Add environment variables
5. ✅ Auto-compile algorithms
6. ✅ Fix silent A* fallback

### Week 2 (Medium)
7. ✅ Add loading states
8. ✅ Show compilation errors
9. ✅ Add error boundary
10. ✅ Validate URDF uploads
11. ✅ Fix path visualization updates

### Week 3 (Low + Cleanup)
12. ✅ Better export errors
13. ✅ Isaac Lab validation
14. ✅ Multi-robot implementation or removal
15. ✅ Increase timeouts

---

## Testing Checklist

After all fixes:

### Functional Tests
- [ ] Chat works without "Sorry, there was an error"
- [ ] Helpful error if Ollama is down
- [ ] Custom algorithms execute successfully
- [ ] Robot follows custom path (not just A*)
- [ ] Path updates when obstacles move
- [ ] URDF validation prevents crashes

### Security Tests
- [ ] No eval() in codebase
- [ ] Input validation on all uploads
- [ ] CORS configured properly

### UX Tests
- [ ] Loading spinners visible during generation
- [ ] Compilation errors shown in UI
- [ ] Error boundary catches Three.js crashes
- [ ] Environment variables work in production

---

**Document Version**: 1.0
**Next Review**: After Week 1 fixes complete
**Owner**: Celestial Studio Team
