%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Epoch & Unix Timestamp Converter
%
%   Definition: Converts an epoch/unix timestamp into a human readable date
%               Output is both a vector of [Y, M, D, H, MN, S] saved in
%               "dv" and a standard date (e.g., 04-Oct-2017 20:00:30) 
%               saved in "y"
%
%   Syntax: [y, dv] = epoch2UTC(epoch)  
%
%   THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
%   WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
%   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
%   NO EVENT SHALL THE COPYRIGHT OWNER BE LIABLE FOR ANY DIRECT, INDIRECT,
%   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
%   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
%   OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
%   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
%   TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
%   USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
%   DAMAGE.
%
%   Written by Dr. Erol Kalkan, P.E. (ekalkan@usgs.gov)
%   URL: www.erolkalkan.com
%   $Revision: 1.0 $  $Date: 2017/05/16 08:37:00 $
function [y, dv] = epoch2UTC(epoch)
dnum = datenum(1970,1,1,0,0,epoch);
dv = datevec(dnum); % date vector as [Y, M, D, H, MN, S]
y = datestr(719529 + epoch/86400,'dd-mmm-yyyy HH:MM:SS');
return